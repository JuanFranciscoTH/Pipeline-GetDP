// ==========================================================================
//  formulation.pro -- GetDP time-harmonic magnetic vector potential ("a")
//  formulation of a stranded circular TX coil over the mine block model.
//  FIXED file (part of the simulator, not a setting): every case-dependent value
//  (regions, sigma / mu_r / eps_r per rock class, coil source, frequency) comes
//  from data.pro, which run.py writes on every run from config.yaml and the CSV.
//
//  Physics (harmonic at Freq_Hz, E = -j w a):
//      curl( nu curl a ) + j w sigma a - w^2 eps a = js0
//    * nu = 1/(mu0 mu_r) per region; eps = eps0 eps_r.
//    * In the conducting rock (DomainC) the medium term is j w (sigma + j w eps) a:
//      conduction (eddy currents / skin depth) PLUS displacement current, with
//      sigma and eps_r of each block (rock class) taken from the CSV.
//    * Air, coil and sigma = 0 rock are magneto-quasistatic (no medium term):
//      valid while the domain is much smaller than the wavelength c/f (60 km at
//      5 kHz).  There a tree (co-tree) gauge removes the gradient null-space of a;
//      in the conducting rock the medium term does it.
//    * js0 = stranded coil current, N*I ampere-turns uniform over the tube section.
//    * Only B = curl a is needed, so no electric scalar potential.
//    * Truncation: a x n = 0 on the outer box (flux tangent to the boundary).
//
//  Run (run.py does it):  getdp formulation.pro -msh model.msh -solve Dyn
//  Output: results/b.pos (B on every element) -> export_vtu.py -> B_field.vtu
// ==========================================================================

Include "data.pro";        // Group{} (regions) + sigma[], mur[], epsr[] + coil constants

Function {
  mu0  = 4.e-7 * Pi;
  eps0 = 8.854187817e-12;
  nu[]  = 1. / (mu0 * mur[]);
  eps[] = eps0 * epsr[];

  // Azimuthal unit vector around the vertical coil axis through (Cx, Cy), times
  // |js| = NI/(pi r^2), so the integral of js over the tube section is NI.
  js0[] = NI_coil / (Pi * Rmin_coil^2)
        * Vector[ -(Y[] - Cy), X[] - Cx, 0. ]
        / Sqrt[ (X[] - Cx)^2 + (Y[] - Cy)^2 + 1.e-30 ];
}

Constraint {
  { Name MVP_Outer; Case { { Region Sur_Outer; Value 0.; } } }
  // spanning tree of the edges in DomainGauge (started on Sur_Outer), a = 0 on it
  { Name Gauge; Type Assign;
    Case { { Region DomainGauge; SubRegion Sur_Outer; Value 0.; } } }
}

Jacobian     { { Name Vol; Case { { Region All; Jacobian Vol; } } } }
Integration  { { Name I1;  Case { { Type Gauss; Case {
                 { GeoElement Tetrahedron; NumberOfPoints 4; }
                 { GeoElement Triangle;    NumberOfPoints 3; } } } } } }

// magnetic vector potential a: edge (Whitney 1-form) elements
FunctionSpace {
  { Name Hcurl_a; Type Form1;
    BasisFunction {
      { Name se; NameOfCoef ae; Function BF_Edge;
        Support Domain; Entity EdgesOf[ All ]; }
    }
    Constraint {
      { NameOfCoef ae; EntityType EdgesOf; NameOfConstraint MVP_Outer; }
      { NameOfCoef ae; EntityType EdgesOfTreeIn; EntitySubType StartingOn;
        NameOfConstraint Gauge; }
    }
  }
}

Formulation {
  { Name MagDyn; Type FemEquation;
    Quantity { { Name a; Type Local; NameOfSpace Hcurl_a; } }
    Equation {
      Galerkin { [ nu[] * Dof{d a}, {d a} ];        In Domain;  Jacobian Vol; Integration I1; }
      // conduction current  j w sigma a
      Galerkin { DtDof[ sigma[] * Dof{a}, {a} ];    In DomainC; Jacobian Vol; Integration I1; }
      // displacement current  (j w)^2 eps a = -w^2 eps a
      Galerkin { DtDtDof[ eps[] * Dof{a}, {a} ];    In DomainC; Jacobian Vol; Integration I1; }
      Galerkin { [ -js0[], {a} ];                   In DomainS; Jacobian Vol; Integration I1; }
    }
  }
}

Resolution {
  { Name Dyn;
    System    { { Name S; NameOfFormulation MagDyn; Type Complex; Frequency Freq_Hz; } }
    Operation { Generate[S]; Solve[S]; SaveSolution[S]; PostOperation[Out_Dyn]; }
  }
}

PostProcessing {
  { Name Post_Dyn; NameOfFormulation MagDyn;
    Quantity { { Name b; Value { Local { [ {d a} ]; In Domain; Jacobian Vol; } } } } }
}

PostOperation {
  { Name Out_Dyn; NameOfPostProcessing Post_Dyn;
    Operation { Print[ b, OnElementsOf Domain, File "results/b.pos" ]; } }
}
