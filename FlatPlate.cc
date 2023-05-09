/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2023 by the deal.II authors
 *
 * Fluid - Structure Interaction
 */
 
 
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
 
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
 
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
/*---------------------------------------------------------------------
 Includes to import the geometry from other softwares such as Gmsh, 
 which we will use in this project
 */
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
 
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
 
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
 
 
#include <fstream>
#include <iostream>
 
namespace FlatPlate
{
  using namespace dealii;
 
 
  template <int dim>
  class StationaryNavierStokes
  {
  public:
    StationaryNavierStokes(const unsigned int degree);
    void run(const unsigned int refinement);
 
  private:
    /*------------------
    Function to extract the Lift and Drag force created on the surface of the desired object
    */
    void compute_lift_and_drag(unsigned int refinement);

    void setup_dofs();
 
    void initialize_system();
 
    void assemble(const bool initial_step, const bool assemble_matrix);
 
    void assemble_system(const bool initial_step);
 
    void assemble_rhs(const bool initial_step);
 
    void solve(const bool initial_step);
 
    void refine_mesh();
 
    void output_results(const unsigned int refinement_cycle) const;
 
    void newton_iteration(const double       tolerance,
                          const unsigned int max_n_line_searches,
                          const unsigned int max_n_refinements,
                          const bool         is_initial_step,
                          const bool         output_result);
 
    void compute_initial_guess(double step_size);
 
    double                               viscosity;
    double                               gamma;
    const unsigned int                   degree;
    std::vector<types::global_dof_index> dofs_per_block;
 
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;
  
 
    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;
 
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    SparseMatrix<double>      pressure_mass_matrix;
 
    BlockVector<double> present_solution;
    BlockVector<double> newton_update;
    BlockVector<double> system_rhs;
    BlockVector<double> evaluation_point;
  };
 
 
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim + 1)
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component) const override;
  };
 
  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    if (component == 0 && std::abs(p[0]) < 1e-10)
      return 1*p[1]*(0.05-p[1]);

    if (component == 0 && std::abs(p[0] - 0.15) < 1e-10)
      return 1*p[1]*(0.05-p[1]);
 
    return 0;
  }
 
  template <class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(double                           gamma,
                             double                           viscosity,
                             const BlockSparseMatrix<double> &S,
                             const SparseMatrix<double> &     P,
                             const PreconditionerMp &         Mppreconditioner);
 
    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;
 
  private:
    const double                     gamma;
    const double                     viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double> &     pressure_mass_matrix;
    const PreconditionerMp &         mp_preconditioner;
    SparseDirectUMFPACK              A_inverse;
  };
 
 
  template <class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                           gamma,
    double                           viscosity,
    const BlockSparseMatrix<double> &S,
    const SparseMatrix<double> &     P,
    const PreconditionerMp &         Mppreconditioner)
    : gamma(gamma)
    , viscosity(viscosity)
    , stokes_matrix(S)
    , pressure_mass_matrix(P)
    , mp_preconditioner(Mppreconditioner)
  {
    A_inverse.initialize(stokes_matrix.block(0, 0));
  }
 
  template <class PreconditionerMp>
  void BlockSchurPreconditioner<PreconditionerMp>::vmult(
    BlockVector<double> &      dst,
    const BlockVector<double> &src) const
  {
    Vector<double> utmp(src.block(0));
 
    {
      SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
      SolverCG<Vector<double>> cg(solver_control);
 
      dst.block(1) = 0.0;
      cg.solve(pressure_mass_matrix,
               dst.block(1),
               src.block(1),
               mp_preconditioner);
      dst.block(1) *= -(viscosity + gamma);
    }
 
    {
      stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
      utmp *= -1.0;
      utmp += src.block(0);
    }
 
    A_inverse.vmult(dst.block(0), utmp);
  }
 
  template <int dim>
  StationaryNavierStokes<dim>::StationaryNavierStokes(const unsigned int degree)
    : viscosity(1.0 / 1000)
    , gamma(1.0)
    , degree(degree)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
  {}
 
  template <int dim>
  void StationaryNavierStokes<dim>::setup_dofs()
  {
    system_matrix.clear();
    pressure_mass_matrix.clear();
 
    dof_handler.distribute_dofs(fe);
 
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);
 
    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];
 
    const FEValuesExtractors::Vector velocities(0);
    {
      nonzero_constraints.clear();
 
      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               ZeroFunction<dim>(dim+1),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
    }
    nonzero_constraints.close();
 
    {
      zero_constraints.clear();
 
      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(
                                                 dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      /* No-Slip boundary condition for the plate*/
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(
                                                 dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
    }
    zero_constraints.close();
 
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << dof_u << " + " << dof_p << ')' << std::endl;
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::initialize_system()
  {
    {
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from(dsp);
    }
 
    system_matrix.reinit(sparsity_pattern);
 
    present_solution.reinit(dofs_per_block);
    newton_update.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::assemble(const bool initial_step,
                                             const bool assemble_matrix)
  {
    if (assemble_matrix)
      system_matrix = 0;
 
    system_rhs = 0;
 
    QGauss<dim> quadrature_formula(degree + 2);
 
    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
 
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
 
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
 
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
 
    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
    std::vector<double>         present_pressure_values(n_q_points);
 
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
 
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
 
        local_matrix = 0;
        local_rhs    = 0;
 
        fe_values[velocities].get_function_values(evaluation_point,
                                                  present_velocity_values);
 
        fe_values[velocities].get_function_gradients(
          evaluation_point, present_velocity_gradients);
 
        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);
 
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                phi_u[k]      = fe_values[velocities].value(k, q);
                phi_p[k]      = fe_values[pressure].value(k, q);
              }
 
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (assemble_matrix)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        local_matrix(i, j) +=
                          (viscosity *
                             scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                           present_velocity_gradients[q] * phi_u[j] * phi_u[i] +
                           grad_phi_u[j] * present_velocity_values[q] *
                             phi_u[i] -
                           div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                           gamma * div_phi_u[j] * div_phi_u[i] +
                           phi_p[i] * phi_p[j]) *
                          fe_values.JxW(q);
                      }
                  }
 
                double present_velocity_divergence =
                  trace(present_velocity_gradients[q]);
                local_rhs(i) +=
                  (-viscosity * scalar_product(present_velocity_gradients[q],
                                               grad_phi_u[i]) -
                   present_velocity_gradients[q] * present_velocity_values[q] *
                     phi_u[i] +
                   present_pressure_values[q] * div_phi_u[i] +
                   present_velocity_divergence * phi_p[i] -
                   gamma * present_velocity_divergence * div_phi_u[i]) *
                  fe_values.JxW(q);
              }
          }
 
        cell->get_dof_indices(local_dof_indices);
 
        const AffineConstraints<double> &constraints_used =
          initial_step ? nonzero_constraints : zero_constraints;
 
        if (assemble_matrix)
          {
            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs);
          }
        else
          {
            constraints_used.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        system_rhs);
          }
      }
 
    if (assemble_matrix)
      {
        pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
        pressure_mass_matrix.copy_from(system_matrix.block(1, 1));
 
        system_matrix.block(1, 1) = 0;
      }
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::assemble_system(const bool initial_step)
  {
    assemble(initial_step, true);
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::assemble_rhs(const bool initial_step)
  {
    assemble(initial_step, false);
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::solve(const bool initial_step)
  {
    const AffineConstraints<double> &constraints_used =
      initial_step ? nonzero_constraints : zero_constraints;
 
    SolverControl solver_control(system_matrix.m(),
                                 1e-4 * system_rhs.l2_norm(),
                                 true);
 
    SolverFGMRES<BlockVector<double>> gmres(solver_control);
    SparseILU<double>                 pmass_preconditioner;
    pmass_preconditioner.initialize(pressure_mass_matrix,
                                    SparseILU<double>::AdditionalData());
 
    const BlockSchurPreconditioner<SparseILU<double>> preconditioner(
      gamma,
      viscosity,
      system_matrix,
      pressure_mass_matrix,
      pmass_preconditioner);
 
    gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
    std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;
 
    constraints_used.distribute(newton_update);
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    const FEValuesExtractors::Vector velocity(0);
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      present_solution,
      estimated_error_per_cell,
      fe.component_mask(velocity));
 
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.0);
 
    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, BlockVector<double>> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
    triangulation.execute_coarsening_and_refinement();
 
    setup_dofs();
 
    BlockVector<double> tmp(dofs_per_block);
 
    solution_transfer.interpolate(present_solution, tmp);
    nonzero_constraints.distribute(tmp);
 
    initialize_system();
    present_solution = tmp;
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::newton_iteration(
    const double       tolerance,
    const unsigned int max_n_line_searches,
    const unsigned int max_n_refinements,
    const bool         is_initial_step,
    const bool         output_result)
  {
    bool first_step = is_initial_step;
 
    for (unsigned int refinement_n = 0; refinement_n < max_n_refinements + 1;
         ++refinement_n)
      {
        unsigned int line_search_n = 0;
        double       last_res      = 1.0;
        double       current_res   = 1.0;
        std::cout << "grid refinements: " << refinement_n << std::endl
                  << "viscosity: " << viscosity << std::endl;
 
        while ((first_step || (current_res > tolerance)) &&
               line_search_n < max_n_line_searches)
          {
            if (first_step)
              {
                setup_dofs();
                initialize_system();
                evaluation_point = present_solution;
                assemble_system(first_step);
                solve(first_step);
                present_solution = newton_update;
                nonzero_constraints.distribute(present_solution);
                first_step       = false;
                evaluation_point = present_solution;
                assemble_rhs(first_step);
                current_res = system_rhs.l2_norm();
                std::cout << "The residual of initial guess is " << current_res
                          << std::endl;
                last_res = current_res;
                std::cout << "For Forces!!!!"<< std::endl;
                compute_lift_and_drag(refinement_n);
              }
            else
              {
                evaluation_point = present_solution;
                assemble_system(first_step);
                solve(first_step);
 
                for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
                  {
                    evaluation_point = present_solution;
                    evaluation_point.add(alpha, newton_update);
                    nonzero_constraints.distribute(evaluation_point);
                    assemble_rhs(first_step);
                    current_res = system_rhs.l2_norm();
                    std::cout << "  alpha: " << std::setw(10) << alpha
                              << std::setw(0) << "  residual: " << current_res
                              << std::endl;
                    std::cout << "Plate Force: \n"<< std::endl;
                    compute_lift_and_drag(refinement_n);
                    if (current_res < last_res)
                      break;
                  }
                {
                  present_solution = evaluation_point;
                  std::cout << "  number of line searches: " << line_search_n
                            << "  residual: " << current_res << std::endl;
                  last_res = current_res;
                }
                ++line_search_n;
                std::cout << "Plate Force: \n"<< std::endl;
                compute_lift_and_drag(refinement_n);
              }
 
            if (output_result)
              {
                output_results(max_n_line_searches * refinement_n +
                               line_search_n);
              }
          }
 
        if (refinement_n < max_n_refinements)
          {
            refine_mesh();
          }
      }
  }
 
  template <int dim>
  void StationaryNavierStokes<dim>::compute_initial_guess(double step_size)
  {
    const double target_Re = 1.0 / viscosity;
 
    bool is_initial_step = true;
 
    for (double Re = 1000.0; Re < target_Re;
         Re        = std::min(Re + step_size, target_Re))
      {
        viscosity = 1.0 / Re;
        std::cout << "Searching for initial guess with Re = " << Re
                  << std::endl;
        newton_iteration(1e-10, 50, 0, is_initial_step, false);
        is_initial_step = false;
      }
  }
 
//-----------------------------------------------------------------------------------------//
/* Regarding the Drag and Lift force function, the first step is to extract the velocity 
 gradient and pressure values. We then integrate these values over the surface of the 
 considered domain, which in this case is the plate surface with ID equal to 2. Next, 
 we extract the fluid stress by solving (1/Re)*velocity and obtain the final fluid 
 stress by subtracting the fluid pressure from (1/Re)*velocity. The total force is 
 obtained by multiplying the stress with the normal vector, face value, and Jacobian
 matrix. The drag force is then the total force in the X direction, while the lift 
 force is the total force in the Y direction. */

  template <int dim>
  void StationaryNavierStokes<dim>::compute_lift_and_drag(unsigned int refinement) {
    QGauss<dim-1> face_quadrature_formula(degree + 2);
    const int n_q_points = face_quadrature_formula.size();
    std::vector<double>                      pressure_values(n_q_points);
    std::vector<SymmetricTensor<2, dim>>     velocity_gradients(n_q_points);
    
    Tensor<1, dim> normal_vector;
    SymmetricTensor<2, dim> fluid_stress;
    SymmetricTensor<2, dim> fluid_pressure;
    Tensor<1, dim> forces;


    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                                update_values | update_quadrature_points | update_gradients |
                                                update_JxW_values | update_normal_vectors);
 
    const FEValuesExtractors::Vector velocity(0);
    const FEValuesExtractors::Scalar pressure(dim);
 
    const double Re = 1.0 / viscosity;
    double Drag = 0.0;
    double Lift = 0.0;

      for(const auto& cell : dof_handler.active_cell_iterators()) {
        if(cell->is_locally_owned()) {
          for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
            if(cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == 2) {
              fe_face_values.reinit(cell, face);
              
              fe_face_values[velocity].get_function_symmetric_gradients(evaluation_point, velocity_gradients); /*--- velocity gradients ---*/
              fe_face_values[pressure].get_function_values(evaluation_point, pressure_values); /*--- pressure values ---*/
  
              for(int q = 0; q < n_q_points; q++) {
                normal_vector = -fe_face_values.normal_vector(q);
  
                for(unsigned int d = 0; d < dim; ++ d) {
                  fluid_pressure[d][d] = pressure_values[q];
                  for(unsigned int k = 0; k < dim; ++k)
                    fluid_stress[d][k] = 1.0/Re*velocity_gradients[q][d][k];
                }
                fluid_stress = fluid_stress - fluid_pressure;
  
                forces = fluid_stress*normal_vector*fe_face_values.JxW(q);
  
                Drag += forces[0];
                Lift += forces[1];
              }
            }
          }
        }
      }

      std::cout << "Lift=" << Lift << std::endl;
      std::cout << "Drag=" << Drag << std::endl;
      std::ofstream f(std::to_string(1.0 / viscosity) + "-line-" +
                  std::to_string(refinement) +"Force" +".txt");
      f << "# Lift  Drag" << std::endl;
      f << Lift;
      f << ' ' << Drag;
      f << std::endl;
  }




//----------------------------------------------------------------------------------------//

  template <int dim>
  void StationaryNavierStokes<dim>::output_results(
    const unsigned int output_index) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
 
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();
 
    std::ofstream output(std::to_string(1.0 / viscosity) + "-solution-" +
                         Utilities::int_to_string(output_index, 4) + ".vtk");
    data_out.write_vtk(output);
    
  }
 
 
 // Mesh
  template <int dim>
  void print_mesh_info(const Triangulation<dim> &triangulation,
                     const std::string &       filename)
  {
    std::cout << "Mesh info:" << std::endl
              << " dimension: " << dim << std::endl
              << " no. of cells: " << triangulation.n_active_cells() << std::endl;
    {
      std::map<types::boundary_id, unsigned int> boundary_count;
      for (const auto &face : triangulation.active_face_iterators())
        if (face->at_boundary())
          boundary_count[face->boundary_id()]++;
       
      std::cout << " boundary indicators: ";
      
      for (const std::pair<const types::boundary_id, unsigned int> &pair :
          boundary_count)
       {
        std::cout << pair.first << '(' << pair.second << " times) ";
       }
      std::cout << std::endl;
         
    }
    std::ofstream out(filename);
    GridOut       grid_out;
    grid_out.write_vtu(triangulation, out);
    std::cout << " written to " << filename << std::endl << std::endl;
  }

  template <int dim>
  void StationaryNavierStokes<dim>::run(const unsigned int refinement)
  {

    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f("Pl20Mesh3dN.msh");
    gridin.read_msh(f);

    triangulation.refine_global(0);

    print_mesh_info(triangulation, "grid-1.vtu");
    DataPostprocessors::BoundaryIds<dim> boundary_ids;
    DataOutFaces<dim> data_out_faces;
    FE_Q<dim>         dummy_fe(1);
 
    DoFHandler<dim>   dummy_dof_handler(triangulation);
    dummy_dof_handler.distribute_dofs(dummy_fe);
 
    Vector<double> dummy_solution (dummy_dof_handler.n_dofs());
 
    data_out_faces.attach_dof_handler(dummy_dof_handler);
    data_out_faces.add_data_vector(dummy_solution, boundary_ids);
    data_out_faces.build_patches();
 
    std::ofstream out("boundary_ids.vtu");
    data_out_faces.write_vtu(out);

    const double Re = 1.0 / viscosity;
    
    
    if (Re > 1000.0)
      {
        std::cout << "Searching for initial guess ..." << std::endl;
        const double step_size = 5000.0;
        compute_initial_guess(step_size);
        std::cout << "Found initial guess." << std::endl;
        std::cout << "Computing solution with target Re = " << Re << std::endl;
        viscosity = 1.0 / Re;
        newton_iteration(1e-10, 50, refinement, false, true);
        
      }
    else
      {
        
        newton_iteration(1e-10, 50, refinement, true, true);
        
      }
  }
} // namespace FlatPlate

int main()
{
  try
    {
      using namespace FlatPlate;
 
      StationaryNavierStokes<3> flow(/* degree = */ 1);
      flow.run(1);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}