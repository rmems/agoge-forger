terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Placeholder for GPU VM
# resource "azurerm_linux_virtual_machine" "agoge_node" {
#   name                = "agoge-forge"
#   resource_group_name = "..."
#   location            = var.location
#   size                = var.vm_size
# }
