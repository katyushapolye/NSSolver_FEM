#include "INSSolver.h"

using namespace dealii;

// Static member definitions

// Mesh and finite elements
Triangulation<2> INSSolver::mesh;
FESystem<2> INSSolver::fe_velocity(FE_Q<2>(INSSolver::K), 2);  // 2 components for velocity
FE_Q<2> INSSolver::fe_pressure(INSSolver::K - 1);

// DoF handlers
DoFHandler<2> INSSolver::dof_handler_velocity(INSSolver::mesh);
DoFHandler<2> INSSolver::dof_handler_pressure(INSSolver::mesh);

// Sparsity patterns and matrices
SparsityPattern INSSolver::velocity_sparsity_pattern;
SparseMatrix<double> INSSolver::velocity_global_mat;
Vector<double> INSSolver::velocity_rhs;

SparsityPattern INSSolver::pressure_sparsity_pattern;
SparseMatrix<double> INSSolver::pressure_global_mat;
Vector<double> INSSolver::pressure_rhs;

// Solution vectors
Vector<double> INSSolver::velocity;
Vector<double> INSSolver::velocity_star;
Vector<double> INSSolver::velocity_old;
Vector<double> INSSolver::p;
Vector<double> INSSolver::p_ant;

AffineConstraints<double> INSSolver::velocity_constraints;
AffineConstraints<double> INSSolver::pressure_constraints;

// Boundary condition helper class
template <int dim>
class VelocityBoundaryValues : public Function<dim>
{
public:
    VelocityBoundaryValues() : Function<dim>(dim) {}
    
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
        if (component == 0) {
            // u_x component
            if (std::abs(p[1] - 1.0) < 1e-10)
                return 1.0;
            return 0.0;
        }
        // u_y component
        return 0.0;
    }
    
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
        values(0) = value(p, 0);
        values(1) = value(p, 1);
    }
};

// Function implementations
void INSSolver::make_grid()
{
    GridGenerator::hyper_cube(mesh, x0, xf);
    mesh.refine_global(5);
    const unsigned int n_boundary_refinements = 1  ;
    const double boundary_distance = 0.1;  // Distance from boundary to refine

    for (unsigned int step = 0; step < n_boundary_refinements; ++step)
    {
        for (auto &cell : mesh.active_cell_iterators())
        {
            // Check if cell is near any boundary
            bool near_boundary = false;
            
            
            // Alternative: refine based on distance from boundary
            Point<2> center = cell->center();
            if (std::abs(center[0]) > 1.0 - boundary_distance ||  std::abs(center[1]) > 1.0 - boundary_distance || std::abs(center[1]) < boundary_distance ||  std::abs(center[0]) < boundary_distance )
            {
                near_boundary = true;
            }
            
            if (near_boundary)
                cell->set_refine_flag();
        }
        
        mesh.execute_coarsening_and_refinement();
    }
    std::cout << "[INSSolver] - Mesh created - active elements: " << mesh.n_active_cells() << std::endl;
}

void INSSolver::setup_system()
{
    // Distribute DoFs
    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);

    std::cout << "[INSSolver] - Velocity DoFs: " << dof_handler_velocity.n_dofs() << std::endl;
    std::cout << "[INSSolver] - Pressure DoFs: " << dof_handler_pressure.n_dofs() << std::endl;

    // Setup velocity constraints
    velocity_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_velocity, velocity_constraints);
    
    // Add Dirichlet BCs for velocity
    VectorTools::interpolate_boundary_values(
        dof_handler_velocity,
        0,
        VelocityBoundaryValues<DIM>(),
        velocity_constraints
    );
    velocity_constraints.close();

    // Setup pressure constraints
    pressure_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler_pressure, pressure_constraints);

    
    // Pin first DOF to zero
    pressure_constraints.add_line(0);
    
    pressure_constraints.close();


    // Setup velocity sparsity pattern and matrix
    DynamicSparsityPattern dsp_velocity(dof_handler_velocity.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp_velocity, velocity_constraints);
    velocity_sparsity_pattern.copy_from(dsp_velocity);
    velocity_global_mat.reinit(velocity_sparsity_pattern);

    // Setup pressure sparsity pattern and matrix
    DynamicSparsityPattern dsp_pressure(dof_handler_pressure.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp_pressure, pressure_constraints);
    pressure_sparsity_pattern.copy_from(dsp_pressure);
    pressure_global_mat.reinit(pressure_sparsity_pattern);

    // Resize vectors
    velocity.reinit(dof_handler_velocity.n_dofs());
    velocity_star.reinit(dof_handler_velocity.n_dofs());
    velocity_old.reinit(dof_handler_velocity.n_dofs());
    velocity_rhs.reinit(dof_handler_velocity.n_dofs());
    
    p.reinit(dof_handler_pressure.n_dofs());
    p_ant.reinit(dof_handler_pressure.n_dofs());
    pressure_rhs.reinit(dof_handler_pressure.n_dofs());
}

void INSSolver::InitializeFEMSolver()
{
    make_grid();
    setup_system();

    double h_min = 1.0 / 128.0;
    double dt_diff = h_min * h_min / (4.0 * viscosity * DIM);
    double dt_conv = h_min / 1.0;
    std::cout << "Suggested dt_max (diffusion): " << dt_diff << std::endl;
    std::cout << "Suggested dt_max (convection): " << dt_conv << std::endl;
}

void INSSolver::StepSimulation(int IT)
{
    assemble_velocity_matrix();
    solve_intermediate_velocity();
    assemble_pressure_matrix();
    solve_pressure();
    project();
    velocity_old = velocity;

    if (IT % 1 == 0)
    {
        output(IT);
    }
}

double INSSolver::getDivSum()
{
    double divsum = 0.0;
    double volume = 0.0;
    
    const QGauss<DIM> quadrature(fe_velocity.degree + 1);
    FEValues<DIM> fe_values(fe_velocity, quadrature,
                            update_gradients | update_JxW_values | update_quadrature_points);
    
    const FEValuesExtractors::Vector velocities(0);
    std::vector<Tensor<2, DIM>> grad_u(quadrature.size());
    
    // Define interior region (avoiding boundaries)
    const double boundary_tolerance = 0.05;  // Stay away from boundaries
    
    for (const auto &cell : dof_handler_velocity.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[velocities].get_function_gradients(velocity, grad_u);
        
        for (unsigned int q = 0; q < quadrature.size(); q++)
        {
            const Point<DIM> &point = fe_values.quadrature_point(q);
            
            // Only compute divergence for interior points
            bool is_interior = true;
            for (unsigned int d = 0; d < DIM; ++d)
            {
                if (point[d] < x0 + boundary_tolerance || 
                    point[d] > xf - boundary_tolerance)
                {
                    is_interior = false;
                    break;
                }
            }
            
            if (is_interior)
            {
                double div = trace(grad_u[q]);
                divsum += std::abs(div) * fe_values.JxW(q);
                volume += fe_values.JxW(q);
            }
        }
    }
    
    // Return average divergence (normalized by interior volume)
    return (volume > 0) ? divsum / volume : 0.0;
}

void INSSolver::assemble_velocity_matrix()
{
    velocity_global_mat = 0;
    velocity_rhs = 0;

    const QGauss<DIM> quadrature(fe_velocity.degree + 1);
    FEValues<DIM> fe_values(fe_velocity, quadrature, 
                            update_quadrature_points | update_values | 
                            update_gradients | update_JxW_values);

    const unsigned int dof_count = fe_velocity.n_dofs_per_cell();
    const unsigned int n_q = quadrature.size();

    FullMatrix<double> local_mat(dof_count, dof_count);
    Vector<double> local_rhs(dof_count);
    std::vector<types::global_dof_index> local_to_global(dof_count);

    // Extractors for vector components
    const FEValuesExtractors::Vector velocities(0);
    
    // Storage for shape function values/gradients
    std::vector<Tensor<1, DIM>> phi_u(dof_count);
    std::vector<Tensor<2, DIM>> grad_phi_u(dof_count);
    std::vector<double> div_phi_u(dof_count);
    
    // Storage for old velocity values
    std::vector<Tensor<1, DIM>> old_velocity_values(n_q);

    for (const auto &cell : dof_handler_velocity.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_to_global);

        local_mat = 0;
        local_rhs = 0;

        // Get old velocity values at quadrature points
        fe_values[velocities].get_function_values(velocity_old, old_velocity_values);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            // Precompute shape function values for this quadrature point
            for (unsigned int k = 0; k < dof_count; ++k)
            {
                phi_u[k] = fe_values[velocities].value(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
            }

            const Tensor<1, DIM> &u_old = old_velocity_values[q];

            for (unsigned int i = 0; i < dof_count; ++i)
            {
                for (unsigned int j = 0; j < dof_count; ++j)
                {
                    // Mass matrix: (phi_i, phi_j)
                    local_mat(i, j) += phi_u[i] * phi_u[j] * fe_values.JxW(q);
                    
                    // Stiffness matrix: nu * dt * (grad phi_i, grad phi_j)
                    local_mat(i, j) += viscosity * dt * 
                                      scalar_product(grad_phi_u[i], grad_phi_u[j]) * 
                                      fe_values.JxW(q);
                    
                    // Convection: dt * (u_old · grad phi_j) * phi_i
                    local_mat(i, j) += dt * (u_old * grad_phi_u[j]) * phi_u[i] * 
                                      fe_values.JxW(q);
                }

                // RHS: M * u_old
                local_rhs(i) += phi_u[i] * u_old * fe_values.JxW(q);
            }
        }

        velocity_constraints.distribute_local_to_global(
            local_mat, local_rhs, local_to_global, 
            velocity_global_mat, velocity_rhs);
    }

    std::cout << "Velocity system assembled!" << std::endl;
}

void INSSolver::assemble_pressure_matrix()
{
    pressure_global_mat = 0.0;
    pressure_rhs = 0.0;

    const QGauss<DIM> quadrature(fe_pressure.degree + 1);
    FEValues<DIM> fe_p(fe_pressure, quadrature, 
                       update_values | update_gradients | update_JxW_values);
    FEValues<DIM> fe_u(fe_velocity, quadrature, update_gradients);

    const unsigned int dofs = fe_pressure.n_dofs_per_cell();
    const unsigned int n_q = quadrature.size();

    FullMatrix<double> local_mat(dofs, dofs);
    Vector<double> local_rhs(dofs);
    std::vector<types::global_dof_index> local_to_global(dofs);

    const FEValuesExtractors::Vector velocities(0);
    std::vector<Tensor<2, DIM>> grad_u_star(n_q);

    auto cell_p = dof_handler_pressure.begin_active();
    auto cell_u = dof_handler_velocity.begin_active();

    for (; cell_p != dof_handler_pressure.end(); ++cell_p, ++cell_u)
    {
        local_mat = 0.0;
        local_rhs = 0.0;
        cell_p->get_dof_indices(local_to_global);

        fe_p.reinit(cell_p);
        fe_u.reinit(cell_u);

        fe_u[velocities].get_function_gradients(velocity_star, grad_u_star);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double div_u_star = trace(grad_u_star[q]);

            for (unsigned int i = 0; i < dofs; ++i)
            {
                for (unsigned int j = 0; j < dofs; ++j)
                {
                    // Laplacian: (grad phi_i, grad phi_j)
                    local_mat(i, j) += fe_p.shape_grad(j, q) * fe_p.shape_grad(i, q) * 
                                      fe_p.JxW(q);
                }

                // RHS: (-1/dt) * div(u_star) * phi_i
                local_rhs(i) += (-1.0 / dt) * div_u_star * fe_p.shape_value(i, q) * 
                               fe_p.JxW(q);
            }
        }

        pressure_constraints.distribute_local_to_global(
            local_mat, local_rhs, local_to_global, 
            pressure_global_mat, pressure_rhs);
    }

    std::cout << "Pressure system assembled!" << std::endl;
}

void INSSolver::solve_intermediate_velocity()
{
    SolverControl solver_control(100000, 1e-12 * velocity_rhs.l2_norm());
    SolverBicgstab<Vector<double>> solver(solver_control);

    solver.solve(velocity_global_mat, velocity_star, velocity_rhs, PreconditionIdentity());
    velocity_constraints.distribute(velocity_star);

    std::cout << solver_control.last_step() 
              << " BiCGStab iterations for intermediate velocity." << std::endl;
}

void INSSolver::solve_pressure()
{
    SolverControl solver_control(100000, 1e-12 * pressure_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);

    solver.solve(pressure_global_mat, p, pressure_rhs, PreconditionIdentity());
    pressure_constraints.distribute(p);

    std::cout << solver_control.last_step() 
              << " CG iterations for pressure." << std::endl;
}

void INSSolver::project()
{
    velocity_global_mat = 0.0;
    velocity_rhs = 0.0;

    const QGauss<DIM> quadrature(fe_velocity.degree + 1);
    FEValues<DIM> fe_v(fe_velocity, quadrature, 
                       update_values | update_gradients | update_JxW_values);
    FEValues<DIM> fe_p(fe_pressure, quadrature, update_values);

    const unsigned int dofs = fe_velocity.n_dofs_per_cell();
    const unsigned int n_q = quadrature.size();

    FullMatrix<double> local_mass(dofs, dofs);
    Vector<double> local_rhs(dofs);
    std::vector<types::global_dof_index> local_dofs(dofs);

    std::vector<double> p_values(n_q);
    const FEValuesExtractors::Vector velocities(0);

    auto cell_v = dof_handler_velocity.begin_active();
    auto cell_p = dof_handler_pressure.begin_active();

    for (; cell_v != dof_handler_velocity.end(); ++cell_v, ++cell_p)
    {
        fe_v.reinit(cell_v);
        fe_p.reinit(cell_p);
        fe_p.get_function_values(p, p_values);

        local_mass = 0.0;
        local_rhs = 0.0;
        cell_v->get_dof_indices(local_dofs);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs; ++i)
            {
                const Tensor<1, DIM> phi_i = fe_v[velocities].value(i, q);
                const Tensor<2, DIM> grad_phi_i = fe_v[velocities].gradient(i, q);

                for (unsigned int j = 0; j < dofs; ++j)
                {
                    const Tensor<1, DIM> phi_j = fe_v[velocities].value(j, q);
                    
                    // Mass matrix
                    local_mass(i, j) += phi_i * phi_j * fe_v.JxW(q);
                }

                // RHS: dt * grad(p) · phi_i
                for (unsigned int d = 0; d < DIM; ++d)
                {
                    local_rhs(i) += dt * p_values[q] * grad_phi_i[d][d] * fe_v.JxW(q);
                }
            }
        }

        velocity_constraints.distribute_local_to_global(
            local_mass, local_rhs, local_dofs, 
            velocity_global_mat, velocity_rhs);
    }

    // Add M·u_star to RHS
    velocity_global_mat.vmult_add(velocity_rhs, velocity_star);

    // Solve
    SolverControl solver_control(100000, 1e-12 * velocity_rhs.l2_norm());
    SolverCG<> solver(solver_control);
    PreconditionJacobi<> preconditioner;
    preconditioner.initialize(velocity_global_mat);

    solver.solve(velocity_global_mat, velocity, velocity_rhs, preconditioner);
    velocity_constraints.distribute(velocity);

    std::cout << "Projection: " << solver_control.last_step() 
              << " CG iterations" << std::endl;
}

void INSSolver::output(int IT)
{
    // Compute divergence per cell
    Vector<double> divergence_per_cell(mesh.n_active_cells());

    QGauss<DIM> quadrature(fe_velocity.degree + 1);
    FEValues<DIM> fe_values(fe_velocity, quadrature,
                            update_gradients | update_JxW_values);

    const FEValuesExtractors::Vector velocities(0);
    std::vector<Tensor<2, DIM>> grad_u(quadrature.size());

    unsigned int cell_index = 0;
    for (const auto &cell : dof_handler_velocity.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values[velocities].get_function_gradients(velocity, grad_u);

        double div_u = 0.0;
        double total_weight = 0.0;

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
            double div_at_q = trace(grad_u[q]);
            div_u += div_at_q * fe_values.JxW(q);
            total_weight += fe_values.JxW(q);
        }

        divergence_per_cell(cell_index) = div_u / total_weight;
        ++cell_index;
    }

    // Output using DataOut
    DataOut<DIM> data_out;
    data_out.attach_dof_handler(dof_handler_velocity);

    // Split velocity into components for visualization
    std::vector<std::string> velocity_names(DIM, "velocity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        velocity_component_interpretation(
            DIM, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(velocity, velocity_names, 
                            DataOut<DIM>::type_dof_data,
                            velocity_component_interpretation);

    data_out.add_data_vector(dof_handler_pressure, p, "pressure");
    data_out.add_data_vector(divergence_per_cell, "divergence");

    data_out.build_patches();

    std::ostringstream filename;
    filename << "Exports/solution_" << std::setfill('0') << std::setw(6) << IT << ".vtk";
    std::ofstream output(filename.str());
    data_out.write_vtk(output);

    std::cout << "Output written for step " << IT << std::endl;
}