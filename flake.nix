{
  description = "inbed: a dependency-free text embedding server (Metal, CUDA, CPU; OpenAI-compatible API)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs { inherit system; config.allowUnfree = true; }));

      inbed = pkgs: extra: pkgs.rustPlatform.buildRustPackage ({
        pname = "inbed";
        version = "0.1.0";
        src = self;
        cargoLock.lockFile = ./Cargo.lock;
        # The golden and kernel tests need model weights and a GPU; run `cargo test` in a checkout.
        doCheck = false;
        meta.mainProgram = "inbed";
      } // extra);
    in
    {
      # `nix run . -- serve --model ...`; `nix build .#cuda` bakes in the NVRTC and cuBLASLt paths.
      packages = forAllSystems (pkgs: {
        default = inbed pkgs { };
      } // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
        cuda = inbed pkgs {
          INBED_CUDA_LIB_PATH = with pkgs.cudaPackages;
            "${pkgs.lib.getLib cuda_nvrtc}/lib:${pkgs.lib.getLib libcublas}/lib";
        };
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [ cargo rustc clippy rustfmt ];
        };
      });
    };
}
