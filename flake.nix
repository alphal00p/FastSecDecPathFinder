{
  description = "Development shell for FastSecDecPathFinder";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv

              # Native build/runtime tools used by pySecDec and OneLOopBridge.
              pkgs.cargo
              pkgs.rustc
              pkgs.pkg-config
              pkgs.gnumake
              pkgs.normaliz
              pkgs.zlib
            ];

            shellHook = ''
              export PYTHON="${python}/bin/python"
              export UV_PYTHON="${python}/bin/python"
              export UV_PYTHON_DOWNLOADS=never
              export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
            '';
          };
        });
    };
}
