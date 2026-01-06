
#include "INSSolver.h"

using namespace dealii;

// Static member definitions

// Mesh and finite elements
Triangulation<2> INSSolver::mesh;
FE_Q<2> INSSolver::fe_velocity(INSSolver::K);
FE_Q<2> INSSolver::fe_pressure(INSSolver::K - 1);

// DoF handlers
DoFHandler<2> INSSolver::dof_handler_velocity(INSSolver::mesh);
DoFHandler<2> INSSolver::dof_handler_pressure(INSSolver::mesh);

// Sparsity patterns and matrices
SparsityPattern INSSolver::velocity_sparsity_pattern;
SparseMatrix<double> INSSolver::velocity_global_mat;
SparseMatrix<double> INSSolver::x_global_mat;
SparseMatrix<double> INSSolver::y_global_mat;
SparsityPattern INSSolver::pressure_sparsity_pattern;
SparseMatrix<double> INSSolver::pressure_global_mat;

// Solution vectors
Vector<double> INSSolver::u_x;
Vector<double> INSSolver::u_y;
Vector<double> INSSolver::u_star_x;
Vector<double> INSSolver::u_star_y;
Vector<double> INSSolver::u_ant_x;
Vector<double> INSSolver::u_ant_y;
Vector<double> INSSolver::x_velocity_rhs;
Vector<double> INSSolver::y_velocity_rhs;
Vector<double> INSSolver::p;
Vector<double> INSSolver::p_ant;
Vector<double> INSSolver::pressure_rhs;

// Function implementations
void INSSolver::make_grid() {
    GridGenerator::hyper_cube(mesh, x0, xf);
    
    mesh.refine_global(5);
    /*
    const unsigned int n_boundary_refinements = 1;
    const double boundary_distance = 0.1;
    for (unsigned int step = 0; step < n_boundary_refinements; ++step)
        {
            for (auto &cell : mesh.active_cell_iterators())
            {
                // Check if cell is near any boundary
                bool near_boundary = false;


                // Alternative: refine based on distance from boundary
                Point<2> center = cell->center();
                if (std::abs(center[0]) > 1.0 - boundary_distance ||
                    std::abs(center[1]) > 1.0 - boundary_distance)
                {
                    near_boundary = true;
                }

                if (near_boundary)
                    cell->set_refine_flag();
            }

            mesh.execute_coarsening_and_refinement();
        }
    */

    std::cout << "[INSSolver] - Mesh created - active elements: "  << mesh.n_active_cells() << std::endl;
}

void INSSolver::setup_system() {
    // Distribute DoFs
    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);
    
    std::cout << "[INSSolver] - Velocity DoFs: " << dof_handler_velocity.n_dofs() << std::endl;
    std::cout << "[INSSolver] - Pressure DoFs: " << dof_handler_pressure.n_dofs() << std::endl;
    
    // Setup velocity sparsity pattern and matrix
    DynamicSparsityPattern dsp_velocity(dof_handler_velocity.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp_velocity);
    velocity_sparsity_pattern.copy_from(dsp_velocity);
    velocity_global_mat.reinit(velocity_sparsity_pattern);
    x_global_mat.reinit(velocity_sparsity_pattern);
    y_global_mat.reinit(velocity_sparsity_pattern);
    // Setup pressure sparsity pattern and matrix
    DynamicSparsityPattern dsp_pressure(dof_handler_pressure.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp_pressure);
    pressure_sparsity_pattern.copy_from(dsp_pressure);
    pressure_global_mat.reinit(pressure_sparsity_pattern);

    
    // Resize vectors
    u_x.reinit(dof_handler_velocity.n_dofs());
    u_y.reinit(dof_handler_velocity.n_dofs());
    x_velocity_rhs.reinit(dof_handler_velocity.n_dofs());
    y_velocity_rhs.reinit(dof_handler_velocity.n_dofs());
    u_star_x.reinit(dof_handler_velocity.n_dofs());
    u_star_y.reinit(dof_handler_velocity.n_dofs());
    u_ant_x.reinit(dof_handler_velocity.n_dofs());
    u_ant_y.reinit(dof_handler_velocity.n_dofs());
    p.reinit(dof_handler_pressure.n_dofs());
    p_ant.reinit(dof_handler_pressure.n_dofs());
    pressure_rhs.reinit(dof_handler_pressure.n_dofs());


}

void INSSolver::InitializeFEMSolver() {



    make_grid();
    setup_system();

    double h_min = 1.0/128.0;
    double dt_diff = h_min * h_min / (4.0 * viscosity * DIM);
    double dt_conv = h_min / 1.0; // Assuming max velocity ~1.0
    std::cout << "Suggested dt_max (diffusion): " << dt_diff << std::endl;
    std::cout << "Suggested dt_max (convection): " << dt_conv << std::endl;
}

void INSSolver::StepSimulation(int IT) {
    assemble_velocity_matrices(INSSolver::u_ant_x,INSSolver::u_ant_y);
    solve_intermediate_velocity();
    assemble_pressure_matrix();
    solve_pressure();
    project();
    INSSolver::u_ant_x = INSSolver::u_x;
    INSSolver::u_ant_y = INSSolver::u_y;
    
    if(IT%1 ==0){
        output(IT);
    }
    


}

double INSSolver::U_X_BOUNDARY_CONDITION(double x, double y) {
    // Return 1.0 on top boundary (y = 1), 0 elsewhere 
    if (std::abs(y - yf) < 1e-10)
        return 1.0;
    return 0.0;
}

double INSSolver::U_Y_BOUNDARY_CONDITION(double x, double y) {
    return 0.0;
}

double INSSolver::P_BOUNDARY_CONDITION(double x, double y) {
    return 0.0;
}

void INSSolver::assemble_velocity_matrices(Vector<double> velocity_x, Vector<double> velocity_y) {

    velocity_global_mat = 0;
    x_velocity_rhs= 0;
    y_velocity_rhs= 0;
    //we will use gauss
    const QGauss<DIM> quadrature(fe_velocity.degree +1);
    FEValues<DIM> fe_values(fe_velocity, quadrature, update_quadrature_points  | update_values | update_gradients | update_JxW_values);
    const unsigned int dof_count = fe_velocity.n_dofs_per_cell();

    //these are the local element mats
    FullMatrix<double> stiffnessMat;    //K
    FullMatrix<double> convectionMat;   //C
    FullMatrix<double> massMat;         //M
    Vector<double> x_rhs;                 
    Vector<double> y_rhs;    
    stiffnessMat.reinit(dof_count, dof_count);
    convectionMat.reinit(dof_count, dof_count);
    massMat.reinit(dof_count, dof_count);
    x_rhs.reinit(dof_count);
    y_rhs.reinit(dof_count);


    std::vector<types::global_dof_index> local_to_global(dof_count); //translation dictorianory from local->global

    for(const auto& cell : dof_handler_velocity.active_cell_iterators()){
        fe_values.reinit(cell);
        cell->get_dof_indices(local_to_global);

        stiffnessMat = 0;
        convectionMat = 0;
        massMat = 0;
        x_rhs = 0; 
        y_rhs = 0;
        for (const unsigned int q : fe_values.quadrature_point_indices()){   //for each quadrature point in our element
            double x = fe_values.quadrature_point(q)[0];//just in case, this is our global coords of this point
            double y = fe_values.quadrature_point(q)[1];//just in case, this is our global coords of this point

            //now, i dont know if this is necessary, i need to double check the maths here!
            // -> in my mind it shouldnt be since this is the previous iteration solution, so just like u0 in the font there isnt any thing we need to do
            // --> I am getting asolutetedly mindmelted by all this localspace global space transformations
            double b_x_q = 0.0, b_y_q = 0.0;
            double u_ant_h = 0.0, v_ant_h = 0.0;
            for (unsigned int k : fe_values.dof_indices()) {
                b_x_q += velocity_x[local_to_global[k]] * fe_values.shape_value(k, q);
                b_y_q += velocity_y[local_to_global[k]] * fe_values.shape_value(k, q);
                u_ant_h += u_ant_x[local_to_global[k]]  * fe_values.shape_value(k, q);
                v_ant_h += u_ant_y[local_to_global[k]]  * fe_values.shape_value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices()){
                for (const unsigned int j : fe_values.dof_indices()){
                    massMat(i,j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
                    stiffnessMat(i,j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
                    convectionMat(i,j) += (b_x_q*fe_values.shape_grad(j, q)[0] + b_y_q*fe_values.shape_grad(j, q)[1])*fe_values.shape_value(i,q)*fe_values.JxW(q);

                }


                x_rhs(i) += fe_values.shape_value(i,q) * (u_ant_h )* fe_values.JxW(q); //if needed, add the font here with x and y 
                y_rhs(i) += fe_values.shape_value(i,q) * (v_ant_h )* fe_values.JxW(q); //or add when the transfer happen
   
            }




        }

        /*local font -> this is the way the tutorialç show, but the above should have the same effect if im not crazy
        //for (const unsigned int i : fe_values.dof_indices()){
        //    for (const unsigned int k : fe_values.dof_indices()){
        //        //x_rhs(i) += massMat(i,k) * u_ant_x[local_to_global[k]];
        //        //y_rhs(i) += massMat(i,k) * u_ant_y[local_to_global[k]];
        //    }
        //}
        */

        //dealii doesnt like doing dt*matrix  like eigen so we do it in parts
        FullMatrix<double> localSystemMat(massMat);
        localSystemMat.add(viscosity * dt, stiffnessMat); //now, because of the part integartion, this is a positive sign, not negative like in FDM
        localSystemMat.add(dt, convectionMat);


        //passing to global matrix assembly 
            for (const unsigned int i : fe_values.dof_indices()){
                for (const unsigned int j : fe_values.dof_indices()){
                    velocity_global_mat.add(local_to_global[i],local_to_global[j], localSystemMat(i, j));
                }

            x_velocity_rhs(local_to_global[i]) += x_rhs(i);
            y_velocity_rhs(local_to_global[i]) += y_rhs(i);
            }

    }

    x_global_mat.copy_from(velocity_global_mat);
    y_global_mat.copy_from(velocity_global_mat);

    std::cout << "System Assembly Completed! - Setting Dirichlet border conditions!";


    std::map<types::global_dof_index,double> u_boundary_values;
    std::map<types::global_dof_index,double> v_boundary_values;
    auto u_boundary_function = [](const Point<DIM>& p){
        return U_X_BOUNDARY_CONDITION(p[0],p[1]);
    };
    auto v_boundary_function = [](const Point<DIM>& p){
        return U_Y_BOUNDARY_CONDITION(p[0],p[1]);
    };

    VectorTools::interpolate_boundary_values(dof_handler_velocity,types::boundary_id(0),ScalarFunctionFromFunctionObject<DIM>(u_boundary_function),u_boundary_values);
    VectorTools::interpolate_boundary_values(dof_handler_velocity,types::boundary_id(0),ScalarFunctionFromFunctionObject<DIM>(v_boundary_function),v_boundary_values);



    MatrixTools::apply_boundary_values(u_boundary_values,x_global_mat,u_star_x,x_velocity_rhs);
    MatrixTools::apply_boundary_values(v_boundary_values,y_global_mat,u_star_y,y_velocity_rhs);

    std::cout << "Border conditions set - Velocity System is ready for solution!\n";


}

void INSSolver::assemble_pressure_matrix() {
    // the equation for pressure laplacian p = 1/dt * div of ustar, note that, u start has a degree of pressure + 1
    pressure_global_mat = 0.0;
    pressure_rhs = 0.0;
    const QGauss<DIM> quadrature(fe_pressure.degree + 1); //this here make sthis works;
    //we use the same quadrature poitns for both fe in pressure and velocity, so the gradient is located at the right place!
    FEValues<DIM> fe_values(fe_pressure,quadrature, update_quadrature_points  | update_values | update_gradients | update_JxW_values);
    FEValues<DIM> fe_values_u(fe_velocity, quadrature, update_gradients);
    const unsigned int dofs = fe_pressure.n_dofs_per_cell();
    const unsigned int dofs_velocity = fe_velocity.n_dofs_per_cell();
    

    FullMatrix<double> local_mat(dofs, dofs);
    Vector<double> local_rhs(dofs);
    std::vector<types::global_dof_index> local_to_global(dofs);
    std::vector<types::global_dof_index> local_to_global_velocity(dofs_velocity);




    std::vector<Tensor<1,DIM>> grad_u_star_x(quadrature.size());
    std::vector<Tensor<1,DIM>> grad_u_star_y(quadrature.size());


    for(const auto& cell : dof_handler_pressure.active_cell_iterators()){ //for each element
        local_mat = 0.0;
        local_rhs = 0.0;
        cell->get_dof_indices(local_to_global);
        
        //we also need the velocity cell, since it lives in  different dof handler
        //deal keeps them consistent as long as we havent messed them independently
        const auto cell_velocity=  typename DoFHandler<DIM>::active_cell_iterator( &mesh,  cell->level(),  cell->index(), &dof_handler_velocity );
        fe_values.reinit(cell);
        fe_values_u.reinit(cell_velocity); //we want the gradient of the u 

        fe_values_u.get_function_gradients(u_star_x, grad_u_star_x);
        fe_values_u.get_function_gradients(u_star_y, grad_u_star_y);

        for(const unsigned int q : fe_values.quadrature_point_indices()){ //for each quadrature point
            double x = fe_values.quadrature_point(q)[0];//just in case, this is our global coords of this point
            double y = fe_values.quadrature_point(q)[1];

            const double div_u_star =  grad_u_star_x[q][0] + grad_u_star_y[q][1]; 


            for(const unsigned int i: fe_values.dof_indices()){
                for(const unsigned int j : fe_values.dof_indices()){
                    //basically a sitffness mat
                    local_mat(i,j) += fe_values.shape_grad(j,q) * fe_values.shape_grad(i,q) * fe_values.JxW(q);
                    

                }
            }

            for (const unsigned int i: fe_values.dof_indices()) {
                local_rhs(i) +=  (1.0 / dt) *  div_u_star *  fe_values.shape_value(i, q) *  fe_values.JxW(q);
                //there maybe a minus here. maybe
            }

        }


        for (const unsigned int i : fe_values.dof_indices()){
        for (const unsigned int j : fe_values.dof_indices()){
                    pressure_global_mat.add(local_to_global[i],local_to_global[j], local_mat(i, j));
        }

            pressure_rhs(local_to_global[i]) += local_rhs(i);

        }


    }

    std::cout << "Pressure system setup completed! - No Dirichlet boundary conditions to be set, rememember to subtract the mean!" << std::endl;


    pressure_global_mat.set(0, 0, 1.0);
    pressure_rhs(0) = 0.0;

    





}

void INSSolver::solve_intermediate_velocity() {
    // Use BiCGStab for non-symmetric systems
    SolverControl solver_control_x(10000, 1e-6 * x_velocity_rhs.l2_norm());
    SolverBicgstab<Vector<double>> solver_x(solver_control_x);
    
    solver_x.solve(x_global_mat, u_star_x, x_velocity_rhs, PreconditionIdentity());
    std::cout << solver_control_x.last_step() << " BiCGStab iterations needed to obtain convergence in u component." << std::endl;
    
    SolverControl solver_control_y(10000, 1e-6 * y_velocity_rhs.l2_norm());
    SolverBicgstab<Vector<double>> solver_y(solver_control_y);
    
    solver_y.solve(y_global_mat, u_star_y, y_velocity_rhs, PreconditionIdentity());
    std::cout << solver_control_y.last_step() << " BiCGStab iterations needed to obtain convergence in v component." << std::endl;
}
void INSSolver::solve_pressure() {
    SolverControl solver_control(10000, 1e-6 * pressure_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(pressure_global_mat, p, pressure_rhs, PreconditionIdentity());
    std::cout << solver_control.last_step() << "  CG iterations needed to obtain convergence in Pressure." << std::endl;
}

void INSSolver::project()
{
    x_velocity_rhs = 0.0; //repurposing
    y_velocity_rhs = 0.0;

    QGauss<DIM> quadrature(fe_velocity.degree + 1);
    FEValues<DIM> fe_v(fe_velocity, quadrature,
                       update_values | update_gradients | update_JxW_values);
    FEValues<DIM> fe_p(fe_pressure, quadrature,
                       update_values);

    const unsigned int dofs_per_cell_v = fe_velocity.dofs_per_cell;
    const unsigned int n_q = quadrature.size();

    Vector<double> local_rhs_x(dofs_per_cell_v);
    Vector<double> local_rhs_y(dofs_per_cell_v);
    std::vector<types::global_dof_index> local_dofs_v(dofs_per_cell_v);

    std::vector<double> p_values(n_q);

    auto cell_v = dof_handler_velocity.begin_active();
    auto cell_p = dof_handler_pressure.begin_active();

    for (; cell_v != dof_handler_velocity.end(); ++cell_v, ++cell_p)
    {
        fe_v.reinit(cell_v);
        fe_p.reinit(cell_p);

        fe_p.get_function_values(p, p_values);

        local_rhs_x = 0.0;
        local_rhs_y = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
            for (unsigned int i = 0; i < dofs_per_cell_v; ++i)
            {
                const Tensor<1,DIM> grad_phi = fe_v.shape_grad(i, q);

                local_rhs_x(i) -= dt * p_values[q] * grad_phi[0] * fe_v.JxW(q);
                local_rhs_y(i) -= dt * p_values[q] * grad_phi[1] * fe_v.JxW(q);
            }

        cell_v->get_dof_indices(local_dofs_v);

        for (unsigned int i = 0; i < dofs_per_cell_v; ++i)
        {
            x_velocity_rhs(local_dofs_v[i]) += local_rhs_x(i);
            y_velocity_rhs(local_dofs_v[i]) += local_rhs_y(i);
        }
    }

    // add M u*
    x_global_mat.vmult_add(x_velocity_rhs, u_star_x);
    y_global_mat.vmult_add(y_velocity_rhs, u_star_y);

    SolverControl solver_control(10000, 1e-6);
    SolverCG<> solver(solver_control);
    PreconditionJacobi<> preconditioner;

    preconditioner.initialize(x_global_mat);
    solver.solve(x_global_mat, u_x, x_velocity_rhs, preconditioner);

    preconditioner.initialize(y_global_mat);
    solver.solve(y_global_mat, u_y, y_velocity_rhs, preconditioner);
}


void INSSolver::output(int IT)
{
    
    ///
    {
        DataOut<DIM> data_out;
        data_out.attach_dof_handler(dof_handler_velocity);

        data_out.add_data_vector(u_x, "u_x");
        data_out.add_data_vector(u_y, "u_y");

        data_out.build_patches();

        std::ostringstream filename;
        filename << "Exports/Velocity/velocity_" << std::setfill('0') << std::setw(6) << IT << ".vtk";

        std::ofstream output(filename.str());
        data_out.write_vtk(output);
    }

    //pre
    {
        DataOut<DIM> data_out;
        data_out.attach_dof_handler(dof_handler_pressure);

        data_out.add_data_vector(p, "p");

        data_out.build_patches();

        std::ostringstream filename;
        filename << "Exports/Pressure/pressure_" << std::setfill('0') << std::setw(6) << IT << ".vtk";

        std::ofstream output(filename.str());
        data_out.write_vtk(output);
    }

    std::cout << "Output written for step " << IT << std::endl;
}
