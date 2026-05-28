terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {}

# Placeholder GPU Droplet
# resource "digitalocean_droplet" "agoge_node" {
#   name   = "agoge-forge"
#   region = var.region
#   size   = var.size
#   image  = "ubuntu-22-04-x64"
# }
