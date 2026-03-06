#include "command_line_params.h"
#include <comm_ggp.h>

// parameters parsed from the command line
//****************************************

// General device/topology/testing options
//----------------------------------------
#ifdef MULTI_GPU
int device_ordinal = -1;
#else
int device_ordinal = 0;
#endif

bool verify_results = false;
bool enable_testing = false;

unsigned long long n_elems = 64;

int rank_order;
std::array<int, 4> gridsize_from_cmdline = {1, 1, 1, 1};
auto &grid_x = gridsize_from_cmdline[0];
auto &grid_y = gridsize_from_cmdline[1];
auto &grid_z = gridsize_from_cmdline[2];
auto &grid_t = gridsize_from_cmdline[3];

bool native_blas_lapack = true;

std::array<int, 4> dim_partitioned = {0, 0, 0, 0};

QudaPrecision prec = QUDA_SINGLE_PRECISION;
QudaPrecision compute_prec = QUDA_SINGLE_PRECISION;
QudaPrecision output_prec = QUDA_SINGLE_PRECISION;
QudaPrecision prec_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision prec_refinement_sloppy = QUDA_INVALID_PRECISION;
QudaPrecision prec_precondition = QUDA_INVALID_PRECISION;
QudaPrecision prec_eigensolver = QUDA_INVALID_PRECISION;
QudaPrecision prec_null = QUDA_INVALID_PRECISION;
QudaPrecision prec_ritz = QUDA_INVALID_PRECISION;
QudaVerbosity verbosity = QUDA_SUMMARIZE;

std::array<int, 4> dim = {24, 24, 24, 24};
std::array<int, 4> grid_partition = {1, 1, 1, 1};

int &xdim = dim[0];
int &ydim = dim[1];
int &zdim = dim[2];
int &tdim = dim[3];

namespace
{
  CLI::TransformPairs<QudaVerbosity> verbosity_map {
    {"silent", QUDA_SILENT}, {"summarize", QUDA_SUMMARIZE}, {"verbose", QUDA_VERBOSE}, {"debug", QUDA_DEBUG_VERBOSE}};

  CLI::TransformPairs<QudaPrecision> precision_map {{"double", QUDA_DOUBLE_PRECISION},
                                                    {"single", QUDA_SINGLE_PRECISION},
                                                    {"half", QUDA_HALF_PRECISION},
                                                    {"quarter", QUDA_QUARTER_PRECISION}};
  
} // namespace

std::shared_ptr<GGPApp> make_app(std::string app_description, std::string app_name)
{
  auto ggp_app = std::make_shared<GGPApp>(app_description, app_name);
  ggp_app->option_defaults()->always_capture_default();

  ggp_app->add_option("--device", device_ordinal, "Set the GPU device to use (default 0, single GPU only)")
    ->check(CLI::Range(0, 16));
  
  ggp_app->add_option("--verbosity", verbosity, "The the verbosity on the top level of QUDA( default summarize)")
    ->transform(CLI::GGPCheckedTransformer(verbosity_map));
  ggp_app->add_option("--verify", verify_results, "Verify the GPU results using CPU results (default true)");

  // lattice dimensions
  auto dimopt = ggp_app->add_option("--dim", dim, "Set space-time dimension (X Y Z T)")->check(CLI::Range(1, 512));
  auto sdimopt = ggp_app
    ->add_option(
		 "--sdim",
		 [](CLI::results_t res) {
		   return CLI::detail::lexical_cast(res[0], xdim) && CLI::detail::lexical_cast(res[0], ydim)
		     && CLI::detail::lexical_cast(res[0], zdim);
		 },
                     "Set space dimension(X/Y/Z) size")
    ->type_name("INT")
    ->check(CLI::Range(1, 512));
  
  ggp_app->add_option("--xdim", xdim, "Set X dimension size(default 24)")
    ->check(CLI::Range(1, 512))
    ->excludes(dimopt)
    ->excludes(sdimopt);
  ggp_app->add_option("--ydim", ydim, "Set X dimension size(default 24)")
    ->check(CLI::Range(1, 512))
    ->excludes(dimopt)
    ->excludes(sdimopt);
  ggp_app->add_option("--zdim", zdim, "Set X dimension size(default 24)")
    ->check(CLI::Range(1, 512))
    ->excludes(dimopt)
    ->excludes(sdimopt);
  ggp_app->add_option("--tdim", tdim, "Set T dimension size(default 24)")->check(CLI::Range(1, 512))->excludes(dimopt);

  // multi-gpu partitioning

  ggp_app->add_option(
    "--partition",
    [](CLI::results_t res) {
      int p;
      auto retval = CLI::detail::lexical_cast(res[0], p);
      for (int j = 0; j < 4; j++) {
        if (p & (1 << j)) { dim_partitioned[j] = 1; }
      }
      return retval;
    },
    "Set the communication topology (X=1, Y=2, Z=4, T=8, and combinations of these)");

  auto gridsizeopt
    = ggp_app
        ->add_option("--gridsize", gridsize_from_cmdline, "Set the grid size in all four dimension (default 1 1 1 1)")
        ->expected(4);
  ggp_app->add_option("--xgridsize", grid_x, "Set grid size in X dimension (default 1)")->excludes(gridsizeopt);
  ggp_app->add_option("--ygridsize", grid_y, "Set grid size in Y dimension (default 1)")->excludes(gridsizeopt);
  ggp_app->add_option("--zgridsize", grid_z, "Set grid size in Z dimension (default 1)")->excludes(gridsizeopt);
  ggp_app->add_option("--tgridsize", grid_t, "Set grid size in T dimension (default 1)")->excludes(gridsizeopt);

  CLI::GGPCheckedTransformer prec_transform(precision_map);
  ggp_app->add_option("--compute-prec", compute_prec, "Precision of data sent to device (default single)")->transform(prec_transform);
  ggp_app->add_option("--output-prec", output_prec, "Precision of result (default single)")->transform(prec_transform);
  //ggp_app->add_option("--prec", prec, "Precision in GPU")->transform(prec_transform);  
  ggp_app->add_option("--n-elems", n_elems, "Number of elements in test (default 64)");
  
  return ggp_app;
  
}

//void add_xengine_option_group(std::shared_ptr<GGPApp> ggp_app) {
//
//}

void add_comms_option_group(std::shared_ptr<GGPApp> ggp_app) {
  auto opgroup
    = ggp_app->add_option_group("Communication", "Options controlling communication (split grid) parameteres");
  opgroup->add_option("--grid-partition", grid_partition, "Set the grid partition (default 1 1 1 1)")->expected(4);
}

void add_testing_option_group(std::shared_ptr<GGPApp> ggp_app) {
  auto opgroup = ggp_app->add_option_group("Testing", "Options controlling automated testing");
  opgroup->add_option("--enable-testing", enable_testing, "Enable automated testing (default false)");
}
