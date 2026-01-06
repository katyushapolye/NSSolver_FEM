#ifndef INS_SOLVER_H
#define INS_SOLVER_H
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_bicgstab.h>  // Add this
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>


#include <string>

using namespace dealii;

//this garbage (probably) implements a Standard Galerkin Finite Element Solver using Pressure Projection for the Incompressible Navier-Stokes Equations
//it delays the non-linear term so I'd guess this makes this a mostly semi-implicit solver, should be reasonably stable for low reynolds regimes and since its FEM
//it should handle boundaries a lot better than the FDM I have
class INSSolver{
public:
    static void InitializeFEMSolver();
    static void StepSimulation(int IT);
    
private:
    static constexpr int DIM = 2;          
    static constexpr int K = 2; //velocity poly degree, should be always >2, pressure is necessarely it k-1
    static constexpr double dt = 0.005;
    static double U_X_BOUNDARY_CONDITION(double x, double y);
    static double U_Y_BOUNDARY_CONDITION(double x, double y);
    static double P_BOUNDARY_CONDITION(double x, double y);
    
    static constexpr double x0 = 0.0;
    static constexpr double xf = 1.0;
    static constexpr double y0 = 0.0;
    static constexpr double yf = 1.0;

    static constexpr double viscosity = 0.01;
    static constexpr double density = 1.0;
    
    static Triangulation<DIM> mesh;
    static FE_Q<DIM> fe_velocity;
    static FE_Q<DIM> fe_pressure;
    
    static DoFHandler<DIM> dof_handler_velocity;
    static DoFHandler<DIM> dof_handler_pressure;
    static SparsityPattern velocity_sparsity_pattern;
    static SparseMatrix<double> velocity_global_mat; //one matrix assembly
    static SparseMatrix<double> x_global_mat; //separate for both components after
    static SparseMatrix<double> y_global_mat; //because of differing BC
    static Vector<double> x_velocity_rhs;
    static Vector<double> y_velocity_rhs;
    static SparsityPattern pressure_sparsity_pattern;
    static SparseMatrix<double> pressure_global_mat;
    static Vector<double> pressure_rhs;
    
    static Vector<double> u_x;
    static Vector<double> u_y;
    static Vector<double> u_star_x;
    static Vector<double> u_star_y;         
    static Vector<double> u_ant_x;
    static Vector<double> u_ant_y;
    static Vector<double> p;
    static Vector<double> p_ant;
    
    static void make_grid();
    static void setup_system();            
    static void assemble_velocity_matrices(Vector<double> u_ant_x, Vector<double> u_ant_y);
    static void assemble_pressure_matrix(); 
    static void solve_intermediate_velocity();
    static void solve_pressure();
    static void project();
    static void output(int IT);
};  

#endif

