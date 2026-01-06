#ifndef INS_SOLVER_H
#define INS_SOLVER_H
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <string>

using namespace dealii;

class INSSolver {
public:
    static void InitializeFEMSolver();
    static void StepSimulation(int IT);
    static double getDivSum();
    
private:
    static constexpr int DIM = 2;          
    static constexpr int K = 2; // velocity poly degree
    static constexpr double dt = 0.005;
    
    static constexpr double x0 = 0.0;
    static constexpr double xf = 1.0;
    static constexpr double y0 = 0.0;
    static constexpr double yf = 1.0;
    static constexpr double viscosity = 0.01;
    static constexpr double density = 1.0;
    
    // Mesh and finite elements
    static Triangulation<DIM> mesh;
    static FESystem<DIM> fe_velocity;  // Vector-valued FE for velocity
    static FE_Q<DIM> fe_pressure;
    
    // DoF handlers
    static DoFHandler<DIM> dof_handler_velocity;
    static DoFHandler<DIM> dof_handler_pressure;
    
    // Matrices and sparsity patterns
    static SparsityPattern velocity_sparsity_pattern;
    static SparseMatrix<double> velocity_global_mat;
    static Vector<double> velocity_rhs;
    
    static SparsityPattern pressure_sparsity_pattern;
    static SparseMatrix<double> pressure_global_mat;
    static Vector<double> pressure_rhs;
    
    // Constraints
    static AffineConstraints<double> velocity_constraints;
    static AffineConstraints<double> pressure_constraints;
    
    // Solution vectors
    static Vector<double> velocity;        // Current velocity (u_x, u_y interleaved)
    static Vector<double> velocity_star;   // Intermediate velocity
    static Vector<double> velocity_old;    // Previous time step
    static Vector<double> p;               // Pressure
    static Vector<double> p_ant;           // Previous pressure
    
    // Boundary condition functions
    static Tensor<1, DIM> velocity_boundary_condition(const Point<DIM> &p);
    static double pressure_boundary_condition(const Point<DIM> &p);
    
    // Assembly and solve functions
    static void make_grid();
    static void setup_system();            
    static void assemble_velocity_matrix();
    static void assemble_pressure_matrix(); 
    static void solve_intermediate_velocity();
    static void solve_pressure();
    static void project();
    static void output(int IT);
};  

#endif