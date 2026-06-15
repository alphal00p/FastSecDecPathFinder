# Subtraction Formula Assets

This directory stores parseable Symbolica expression JSON files for universal
endpoint-projector and regular-Taylor formula signatures.

The formula signatures do not depend on a topology, masses, kinematics, or
sector variable names.  A curated subset can therefore be treated as part of
FSD itself and shipped with the code when it gives a clear runtime benefit.

Generated exploratory cache JSON files are written in this directory root and
are ignored by git by default.  Curated files intended to ship with FSD should
be placed in `curated/`; the loader checks both locations and prefers curated
assets over local generated files with the same signature.

Curated assets are source code for the purpose of FSD behavior.  A high-axis or
large-volume regular-Taylor signature that would otherwise be skipped by the
cold-build guard is prepared by default when a matching curated JSON file is
present.  The full local triple-box cache can be hundreds of MB, and some
high-axis direct formulae are currently slower per sample than the guarded
sparse fallback.  Promote only validated cache files deliberately.
