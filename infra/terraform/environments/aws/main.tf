terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# TODO: Remote state backend
# backend "remote" { ... }

# Placeholder for GPU Instance
# resource "aws_instance" "agoge_training_node" {
#   ami           = "ami-..." # Deep Learning AMI
#   instance_type = var.instance_type
#   
#   tags = {
#     Name = "agoge-forge"
#   }
# }
