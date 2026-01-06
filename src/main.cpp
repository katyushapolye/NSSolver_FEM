#include "headers/WaveSolver.h"
#include "headers/INSSolver.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <implot3d.h>

int main()
{

  if (!glfwInit())
  {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  // GL 3.3 + GLSL 330
  const char *glsl_version = "#version 330";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWmonitor *monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode *mode = glfwGetVideoMode(monitor);

  GLFWwindow *window = glfwCreateWindow(800, 800, "Wave Solver", nullptr, nullptr);
  glfwSetWindowPos(window, 0, 0);

  if (window == nullptr)
  {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); //

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImPlot3D::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  ImGui::StyleColorsDark();

  //WaveSolver solver;
  //solver.init();

  INSSolver::InitializeFEMSolver();

  std::vector<double> time;
  std::vector<double> energy;

  double lastFrameTime = glfwGetTime();
  const double targetFrameTime = 1.0 / 100.0;
  unsigned int IT = 0;

  while (!glfwWindowShouldClose(window))
  {
    IT++;

    // Frame rate limiting
    double currentTime = glfwGetTime();
    double deltaTime = currentTime - lastFrameTime;

    if (deltaTime < targetFrameTime)
    {
      // Sleep for the remaining time
      double sleepTime = targetFrameTime - deltaTime;
      std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
      currentTime = glfwGetTime();
      deltaTime = currentTime - lastFrameTime;
    }

    lastFrameTime = currentTime;

    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //solver.step();
 
    INSSolver::StepSimulation(IT);

    /*
    //ImGui::Text("Current Time: %f s", solver.getCurrentTime());
    //ImGui::SameLine();
    //ImGui::Text("Current Energy: %f j", solver.getEnergy());

    //time.push_back(solver.getCurrentTime());
    //energy.push_back(solver.getEnergy());

    auto surf = solver.extract_surface();

    // 3D Surface Plot
    ImPlot3D::PushColormap("Jet");

    if (ImPlot3D::BeginPlot("Wave Surface"))
    {
      ImPlot3D::SetupAxesLimits(-1, 1, -1, 1, -1, 1);

      float z_min = *std::min_element(surf.quad_colors.begin(), surf.quad_colors.end());
      float z_max = *std::max_element(surf.quad_colors.begin(), surf.quad_colors.end());

      for (size_t i = 0; i < surf.quad_colors.size(); ++i)
      {
        float t = (surf.quad_colors[i] - z_min) / (z_max - z_min);
        ImVec4 color = ImPlot3D::SampleColormap(t);
        ImPlot3D::SetNextFillStyle(color);
        ImPlot3D::PlotQuad("", &surf.x[i * 4], &surf.y[i * 4], &surf.z[i * 4], 4, ImPlot3DQuadFlags_NoMarkers);
      }
      ImPlot3D::EndPlot();
    }
    ImPlot3D::PopColormap();

    // Energy Time Series Plot
    if (ImPlot::BeginPlot("Energy vs Time", ImVec2(-1, 200)))
    {
      // Set up axes
      double current_time = solver.getCurrentTime();
      double window_size = 5.0; // Show ±5 seconds from current time

      ImPlot::SetupAxes("Time (s)", "Energy (J)");
      ImPlot::SetupAxisLimits(ImAxis_X1, current_time - window_size, current_time + window_size, ImGuiCond_Always);

      // Auto-fit Y axis based on visible data
      if (!energy.empty())
      {
        double e_min = *std::min_element(energy.begin(), energy.end());
        double e_max = *std::max_element(energy.begin(), energy.end());
        double margin = (e_max - e_min) * 0.1; // 10% margin
        ImPlot::SetupAxisLimits(ImAxis_Y1, e_min - margin, e_max + margin, ImGuiCond_Always);
      }

      // Plot the energy line
      if (time.size() > 1)
      {
        ImPlot::PlotLine("Energy", time.data(), energy.data(), time.size());
      }

      ImPlot::EndPlot();
    }
      */

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImPlot3D::DestroyContext();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}