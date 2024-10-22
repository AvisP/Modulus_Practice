import modulus
import modulus.sym
from modulus.sym.hydra import to_yaml
from modulus.sym.hydra.utils import compose
from modulus.sym.hydra.config import ModulusConfig
from modulus.models.mlp.fully_connected import FullyConnected
import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from BioReactor import Bioreactor

# from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.key import Key
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter
)

from modulus.sym.geometry.primitives_3d import Cylinder
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
  # make list of nodes to unroll graph on
  br = Bioreactor()
  flow_net = instantiate_arch(
        input_keys=[Key("z"), Key("r"), Key("t")],
        output_keys=[Key("v_z")],
        cfg=cfg.arch.fully_connected,
  )

# flow_net = FullyConnectedArch(
#     input_keys=[Key("z"), Key("r"), Key("t")],
#     output_keys=[Key("v_z")],
#     layer_size=512,
#     nr_layers=3,
#     skip_connections=True,
#     activation_fn=Activation.SILU,
#     adaptive_activations=False,
#     weight_norm=True,
# )  
  nodes = br.make_nodes() + [flow_net.make_node(name="flow_network")]
  r, z, t_symbol = Symbol("x"), Symbol("t")
  radius = 0.0255
  height = 0.3
  time_range = {t_symbol: (0, 2 * 2)}
  cylinder = Cylinder(center=(0, 0, 0), radius=radius, height=height)

  # make domain
  domain = Domain()

  BC = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=cylinder,
    outvar={"v_z": 0, "t":0},
    # batch_size=cfg.batch_size.BC,
    batch_size = 1000, 
    parameterization=time_range,
  )
  domain.add_constraint(BC, "Boundary")


  # interior
  IC = PointwiseInteriorConstraint(
      nodes=nodes,
      geometry=cylinder,
      outvar={"reynold_eq": 0, "material_balance_eq": 0},
      # batch_size=cfg.batch_size.interior,
      batch_size = 4000, 
      parameterization=time_range,
  )
  domain.add_constraint(IC, "interior")

  slv = Solver(cfg, domain)

  slv.solve()

if __name__ == "__main__":
    run()
