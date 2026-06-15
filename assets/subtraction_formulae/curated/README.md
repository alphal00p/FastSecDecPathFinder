# Curated Subtraction Formula Assets

Files in this directory are treated as part of FSD itself.  They are universal
for their endpoint-projector or regular-Taylor signature and are preferred over
locally generated exploratory cache files with the same signature.

The shipped set currently contains the validated endpoint-projector signatures
and the first small validated regular-Taylor signatures used by the
`PSD213`-class triple-box sectors.  Those regular-Taylor files are intentionally
small and evaluator-dominated; larger exploratory regular-Taylor files remain
outside the curated source set until they show a runtime benefit.

Only promote a generated JSON file here after checking both cold-generation cost
and hot-path runtime.  A curated regular-Taylor asset bypasses the default
axis/volume guard, so an overly large direct formula can make ordinary runs
slower even though it removes Python fallback work.
