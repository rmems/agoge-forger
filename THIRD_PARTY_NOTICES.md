# Third-party licensing and notices

The Apache-2.0 license in [`LICENSE`](LICENSE) applies to Agoge Forger's own
source and documentation. It does not relicense Python dependencies, base
models, datasets, adapters, tokenizer assets, or generated model weights.

No third-party source or model artifact is vendored in this repository at the
time this guidance was written, and this file does not claim that any particular
upstream license applies. Resolve and record licenses from authoritative
upstream metadata for the exact revisions used in a distribution.

## Python dependencies

Before distributing a wheel, container, or bundled environment:

1. resolve the exact dependency set from `pyproject.toml` and `uv.lock`;
2. inspect each installed distribution's license metadata and included license
   files rather than inferring a license from its name;
3. preserve copyright, attribution, license, source-offer, and NOTICE material
   required by those exact versions; and
4. review the resulting inventory whenever the lockfile changes.

The dependency declarations and lockfile are reproducibility inputs, not a
license inventory. Automated scanners can assist discovery, but ambiguous or
missing metadata requires verification against the upstream release.

## Models, datasets, and generated artifacts

Before downloading, training on, publishing, or redistributing an artifact:

- pin the model and dataset revisions;
- save the upstream model card, dataset card, and license identifiers with the
  run provenance;
- confirm that the intended use and redistribution are permitted;
- preserve required attribution and use restrictions in the release materials;
- review whether adapter, merged-weight, and generated-data terms differ; and
- do not label an output Apache-2.0 merely because Agoge's source code uses that
  license.

## NOTICE handling

Agoge does not currently ship a root `NOTICE` file because no verified,
project-level third-party NOTICE text has been identified for inclusion. Do not
create speculative attribution text.

When copied or modified Apache-licensed material includes a NOTICE file, carry
forward the relevant notices as required by section 4(d) of Apache-2.0. When a
future distribution bundles third-party material, add only verified notices and
identify their source and revision. NOTICE text is informational and must not be
used to alter an upstream license.
