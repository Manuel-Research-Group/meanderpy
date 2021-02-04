library(reticulate)
library(rayshader)

np <- import("numpy")

grid <- 25
terrain <- np$load("C:/Users/beure/Desktop/TCC/meanderpy/meanderpy/terrain.npy")
shape <- dim(terrain)

basin <- terrain[,,shape[3]]

#We use another one of rayshader's built-in textures:
basin %>%
  sphere_shade(texture = "imhof4") %>%
  add_shadow(ray_shade(basin, zscale = grid), 0.5) %>%
  add_shadow(ambient_shade(basin), 0) %>%
  plot_3d(basin, zscale = grid, fov = 0, theta = 7.5, zoom = 0.5, phi = 15, windowsize = c(1000, 800))
