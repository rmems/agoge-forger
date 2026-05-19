terraform {
  required_providers {
    ibm = {
      source  = "IBM-Cloud/ibm"
      version = "~> 1.50"
    }
  }
}

provider "ibm" {
  region = var.region
}

# Placeholder Bare Metal GPU
# resource "ibm_is_bare_metal_server" "agoge_node" {
#   name    = "agoge-forge"
#   profile = "bx2d-metal-96x384" # Change to GPU profile
# }
