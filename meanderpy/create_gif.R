library(reticulate)
library(rayshader)
library(magick)

np <- import("numpy")

grid <- 25
terrain <- np$load("C:\Users\beure\OneDrive\Documentos\TCC\meanderpy\meanderpy/terrain.npy")
shape <- dim(terrain)
z <- shape[3]

img_frames <- paste0("drain", seq_len(z), ".png")


print(z)
for (i in seq_len(z)) {
   basin <- terrain[,,i]
   print(i)
   basin %>%
     sphere_shade(texture = "imhof4") %>%
     add_shadow(ray_shade(basin, zscale = grid), 0.5) %>%
     add_shadow(ambient_shade(basin), 0) %>%
     plot_3d(basin, zscale = grid, fov = 0, theta = 7.5, zoom = 0.5, 
             phi = 15, windowsize = c(1000, 800))
   
   render_snapshot(img_frames[i])
   rgl::clear3d()
}


image_write_gif(
  image_read(img_frames), 
  path = "C:\Users\beure\OneDrive\Documentos\TCC\meanderpy\meanderpy\evolution15.gif", 
  delay = 10/z
)