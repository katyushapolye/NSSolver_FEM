#ifndef WAVE_SOLVER_H
#define WAVE_SOLVER_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h> //lagrange elements
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/numerics/data_out.h>
#include <string>

using namespace dealii;

struct SurfaceData
{
    std::vector<float> x, y, z;
    std::vector<float> quad_colors; // One per quad
};

class WaveSolver
{
public:
    WaveSolver();
    WaveSolver(double x0, double xf);
    void step();
    void init();
    SurfaceData extract_surface();

    double getCurrentTime();
    unsigned int getIT();
    double getEnergy();

private:
    double currentTime;
    unsigned int IT;
    double t0, tF;
    const double dt = 0.01;
    const double k = 1;
    const double c = 1.0;
    double x0, xf;

    Triangulation<2> mesh;
    FE_Q<2> fe;
    DoFHandler<2> dof_handler;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> global_mat;
    Vector<double> global_sol;
    Vector<double> u0;
    Vector<double> u1;
    Vector<double> global_rhs;

    // used in energy calculation
    SparseMatrix<double> K_global;
    SparseMatrix<double> M_global;

    void make_grid();
    void setup_system();
    void assemble_matrices();
    void assemble_system(Vector<double> u1, Vector<double> u0, double current_time);
    void assemble_system_nonconserving(Vector<double> u1, Vector<double> u0, double current_time);
    void solve();
    void output(int IT);

    double F(double x, double y);
    double G(double x, double y, double t);
};
#endif

double WaveSolver::F(double x, double y)
{
    return 0.0; // sin(3.1415/2.0*x)*sin(3.1415/2.0*y);
}

double WaveSolver::G(double x, double y, double t)
{
    if (t < 0.5 && x <= -1.0 && y > -0.333 && y < 0.33)
    {
        return sin(4 * 3.1415 * t);
    }
    else
    {
        return 0.0;
    }
}

WaveSolver::WaveSolver() : fe(k), dof_handler(mesh)
{
    this->x0 = -1.0;
    this->xf = 1.0;
    this->t0 = 0.0;
    this->tF = 5.0;
}

WaveSolver::WaveSolver(double x0_, double xf_) : fe(k), dof_handler(mesh)
{
    this->x0 = x0_;
    this->xf = xf_;
    this->t0 = 0.0;
    this->tF = 5.0;
}

void WaveSolver::make_grid()
{
    GridGenerator::hyper_cube(mesh, -1.0, 1.0);
    mesh.refine_global(4);
    
    // Local refinement near boundaries
    const unsigned int n_boundary_refinements = 2;
    const double boundary_distance = 0.2;  // Distance from boundary to refine
    
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
    
    std::cout << "Mesh created - active elements: " << mesh.n_active_cells() << std::endl;


    
    // Export mesh as SVG
    std::ofstream out("mesh.svg");
    GridOut grid_out;
    GridOutFlags::Svg svg_flags;
    
    // Optional: customize SVG output
    svg_flags.line_thickness = 2;
    svg_flags.boundary_line_thickness = 4;
    svg_flags.coloring = GridOutFlags::Svg::subdomain_id;  // or level_number, material_id
    
    grid_out.set_flags(svg_flags);
    grid_out.write_svg(mesh, out);
    
    std::cout << "Mesh exported to mesh.svg" << std::endl;
}
void WaveSolver::setup_system()
{
    this->dof_handler.distribute_dofs(this->fe);
    std::cout << "DoF's distributed -  Number of DoFs: " << this->dof_handler.n_dofs() << std::endl;

    DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(this->dof_handler, dsp);
    this->sparsity_pattern.copy_from(dsp);

    this->global_mat.reinit(this->sparsity_pattern);
    this->global_rhs.reinit(this->dof_handler.n_dofs());
    this->global_sol.reinit(this->dof_handler.n_dofs());
    this->u0.reinit(this->dof_handler.n_dofs());
    this->u1.reinit(this->dof_handler.n_dofs());
    this->M_global.reinit(this->sparsity_pattern);
    this->K_global.reinit(this->sparsity_pattern);

    VectorTools::interpolate(dof_handler,
                             ScalarFunctionFromFunctionObject<2>([this](const Point<2> &p)
                                                                 { return this->F(p[0], p[1]); }),
                             this->u0);
    VectorTools::interpolate(dof_handler,
                             ScalarFunctionFromFunctionObject<2>([this](const Point<2> &p)
                                                                 { return this->F(p[0], p[1]); }),
                             this->u1);

    std::cout << "Global Vector and Matrix has been setup!" << std::endl;
}

void WaveSolver::assemble_matrices()
{
    // Assemble M and K for energy calculation
    const QGauss<2> quadrature_formula(fe.degree + 1);
    FEValues<2> fe_values(fe, quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

    const unsigned int dofs = fe.n_dofs_per_cell();
    FullMatrix<double> local_mass(dofs, dofs);
    FullMatrix<double> local_stiff(dofs, dofs);
    std::vector<types::global_dof_index> local_dof_indices(dofs);

    M_global = 0.0;
    K_global = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_mass = 0.0;
        local_stiff = 0.0;
        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                {
                    local_mass(i, j) += fe_values.shape_value(i, q_index) *
                                        fe_values.shape_value(j, q_index) *
                                        fe_values.JxW(q_index);

                    local_stiff(i, j) += fe_values.shape_grad(i, q_index) *
                                         fe_values.shape_grad(j, q_index) *
                                         fe_values.JxW(q_index);
                }
            }
        }

        // Assemble into global
        for (const unsigned int i : fe_values.dof_indices())
        {
            for (const unsigned int j : fe_values.dof_indices())
            {
                M_global.add(local_dof_indices[i], local_dof_indices[j], local_mass(i, j));
                K_global.add(local_dof_indices[i], local_dof_indices[j], local_stiff(i, j));
            }
        }
    }

    // Scale K by c²
    K_global *= (c * c);

    std::cout << "Mass and Stiffness matrices assembled for energy calculation!" << std::endl;
}

void WaveSolver::assemble_system(Vector<double> u1, Vector<double> u0, double current_time)
{
    global_mat = 0;
    global_rhs = 0;

    const QGauss<2> quadrature_formula(fe.degree + 1);
    FEValues<2> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points);
    const unsigned int dofs = fe.n_dofs_per_cell();

    FullMatrix<double> local_mat(dofs, dofs);
    Vector<double> local_rhs(dofs);
    std::vector<types::global_dof_index> local_dof_indices(dofs);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_mat = 0.0;
        local_rhs = 0.0;
        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            double x = fe_values.quadrature_point(q_index)[0];
            double y = fe_values.quadrature_point(q_index)[1];

            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                {
                    double m_ij = fe_values.shape_value(i, q_index) *
                                  fe_values.shape_value(j, q_index) *
                                  fe_values.JxW(q_index);
                    double k_ij = fe_values.shape_grad(i, q_index) *
                                  fe_values.shape_grad(j, q_index) *
                                  fe_values.JxW(q_index);
                    local_mat(i, j) += m_ij + 0.25 * dt * dt * c * c * k_ij;
                }
            }

            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                {
                    double m_ij = fe_values.shape_value(i, q_index) *
                                  fe_values.shape_value(j, q_index) *
                                  fe_values.JxW(q_index);
                    double k_ij = fe_values.shape_grad(i, q_index) *
                                  fe_values.shape_grad(j, q_index) *
                                  fe_values.JxW(q_index);

                    local_rhs(i) += (2.0 * u1(local_dof_indices[j]) - 1.0 * u0(local_dof_indices[j])) *
                                    fe_values.shape_value(j, q_index) *
                                    fe_values.shape_value(i, q_index) *
                                    fe_values.JxW(q_index);

                    local_rhs(i) -= 0.25 * dt * dt * c * c * k_ij * u0(local_dof_indices[j]);
                }

                local_rhs(i) += dt * dt * F(x, y) *
                                fe_values.shape_value(i, q_index) *
                                fe_values.JxW(q_index);
            }
        }

        for (const unsigned int i : fe_values.dof_indices())
        {
            for (const unsigned int j : fe_values.dof_indices())
            {
                global_mat.add(local_dof_indices[i], local_dof_indices[j], local_mat(i, j));
            }
        }
        for (const unsigned int i : fe_values.dof_indices())
        {
            global_rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }

    std::cout << "System Assembly completed -  Setting border conditions...\n";

    std::map<types::global_dof_index, double> boundary_values;

    auto boundary_func = [this, current_time](const Point<2> &p)
    {
        return this->G(p[0], p[1], current_time);
    };
    VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(0),
                                             ScalarFunctionFromFunctionObject<2>(boundary_func),
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values, global_mat, global_sol, global_rhs);

    std::cout << "Border conditions set - System is ready for solution!\n";
}

void WaveSolver::assemble_system_nonconserving(Vector<double> u1, Vector<double> u0, double current_time)
{
    global_mat = 0;
    global_rhs = 0;

    const QGauss<2> quadrature_formula(fe.degree + 1);
    FEValues<2> fe_values(fe, quadrature_formula, update_values | update_gradients | update_JxW_values | update_quadrature_points);
    const unsigned int dofs = fe.n_dofs_per_cell();

    FullMatrix<double> local_mat(dofs, dofs);
    Vector<double> local_rhs(dofs);
    std::vector<types::global_dof_index> local_dof_indices(dofs);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_mat = 0.0;
        local_rhs = 0.0;
        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            double x = fe_values.quadrature_point(q_index)[0];
            double y = fe_values.quadrature_point(q_index)[1];

            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                {
                    double m_ij = fe_values.shape_value(i, q_index) *
                                  fe_values.shape_value(j, q_index) *
                                  fe_values.JxW(q_index);
                    double k_ij = fe_values.shape_grad(i, q_index) *
                                  fe_values.shape_grad(j, q_index) *
                                  fe_values.JxW(q_index);
                    local_mat(i, j) += m_ij + dt * dt * c * c * k_ij;
                }
            }

            for (const unsigned int i : fe_values.dof_indices())
            {
                for (const unsigned int j : fe_values.dof_indices())
                {
                    local_rhs(i) += (2.0 * u1(local_dof_indices[j]) - 1.0 * u0(local_dof_indices[j])) *
                                    fe_values.shape_value(j, q_index) *
                                    fe_values.shape_value(i, q_index) *
                                    fe_values.JxW(q_index);
                }

                local_rhs(i) += dt * dt * F(x, y) *
                                fe_values.shape_value(i, q_index) *
                                fe_values.JxW(q_index);
            }
        }

        for (const unsigned int i : fe_values.dof_indices())
        {
            for (const unsigned int j : fe_values.dof_indices())
            {
                global_mat.add(local_dof_indices[i], local_dof_indices[j], local_mat(i, j));
            }
        }
        for (const unsigned int i : fe_values.dof_indices())
        {
            global_rhs(local_dof_indices[i]) += local_rhs(i);
        }
    }

    std::cout << "System Assembly completed -  Setting border conditions...\n";

    std::map<types::global_dof_index, double> boundary_values;

    auto boundary_func = [this, current_time](const Point<2> &p)
    {
        return this->G(p[0], p[1], current_time);
    };
    VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(0),
                                             ScalarFunctionFromFunctionObject<2>(boundary_func),
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values, global_mat, global_sol, global_rhs);

    std::cout << "Border conditions set - System is ready for solution!\n";
}

void WaveSolver::solve()
{
    SolverControl solver_control(1000, 1e-6 * global_rhs.l2_norm());
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(global_mat, global_sol, global_rhs, PreconditionIdentity());
    std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
}

void WaveSolver::output(int IT)
{
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(global_sol, "solution");
    data_out.build_patches();

    const std::string filename = "solution_" + std::to_string(IT) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    std::cout << "Output written to " << filename << std::endl;
}

void WaveSolver::init()
{
    this->make_grid();
    this->setup_system();
    this->assemble_matrices(); // Assemble M and K once for energy calculation
    currentTime = dt;
    IT = 1;
}

void WaveSolver::step()
{
    this->assemble_system(this->u1, this->u0, currentTime);
    this->solve();
    this->u0 = this->u1;
    this->u1 = this->global_sol;
    currentTime += dt;
    IT++;
}

unsigned int WaveSolver::getIT()
{
    return IT;
}

double WaveSolver::getCurrentTime()
{
    return currentTime;
}

SurfaceData WaveSolver::extract_surface()
{
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(global_sol, "u");
    data_out.build_patches();

    SurfaceData surf;

    for (const auto &patch : data_out.get_patches())
    {
        const unsigned int n_subdivisions = patch.n_subdivisions;
        const unsigned int n = n_subdivisions + 1;

        for (unsigned int i = 0; i < n_subdivisions; ++i)
        {
            for (unsigned int j = 0; j < n_subdivisions; ++j)
            {
                unsigned int idx[4] = {
                    i * n + j,
                    i * n + (j + 1),
                    (i + 1) * n + (j + 1),
                    (i + 1) * n + j};

                float avg_z = 0.0f;
                for (int k = 0; k < 4; ++k)
                {
                    surf.x.push_back((float)patch.vertices[idx[k]][0]);
                    surf.y.push_back((float)patch.vertices[idx[k]][1]);
                    float z = (float)patch.data(0, idx[k]);
                    surf.z.push_back(z);
                    avg_z += z;
                }
                avg_z /= 4.0f;
                surf.quad_colors.push_back(avg_z);
            }
        }
    }

    return surf;
}

double WaveSolver::getEnergy()
{
    // Calculate velocity: v = (u1 - u0) / dt
    Vector<double> velocity(u1.size());
    velocity = u1;
    velocity.add(-1.0, u0);
    velocity *= (1.0 / dt);

    // kinetic energy: (1/2) * v^T * M * v
    double kinetic_energy = M_global.matrix_norm_square(velocity) / 2.0;

    // potential energy: (1/2) * u^T * K * u
    double potential_energy = K_global.matrix_norm_square(u1) / 2.0;

    return kinetic_energy + potential_energy;
}