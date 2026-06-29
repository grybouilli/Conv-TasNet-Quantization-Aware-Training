{
  description = "Impure nix flake for HATorch";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.cudaSupport = true;
            config.allowUnfree = true; # Unfree cause of CUDA
          };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python3
              uv
            ];

            shellHook = ''
              unset PYTHONPATH
              uv sync --extra cu130 --extra dev
              . .venv/bin/activate
            '';

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              "/run/opengl-driver/"
            ];
          };
        }
      );
    };
}
