val_generator <- function(image_paths, 
                          samples_index,
                          mask_paths, 
                          batch_size) {
  
  images_iter <- image_paths[samples_index] # for current epoch
  images_all <- image_paths[samples_index]  # for next epoch
  masks_iter <- mask_paths[samples_index] # for current epoch
  masks_all <- mask_paths[samples_index] # for next epoch
  
  function() {
    
    library(purrr)
    
    cl <- makePSOCKcluster(names = 2) # doParallel  
    
    clusterEvalQ(cl, {
      library(magick)     
      library(abind)     
      library(reticulate)
      
      imagesRead <- function(image_file,
                             mask_file,
                             target_width = 256, 
                             target_height = 256) {
        
        img <- image_read(image_file)
        
        #---following not currently used - code for border
        # img_current_width <-  unname(image_info(img)[[2]])
        # img_current_height <-  unname(image_info(img)[[3]])
        # img_diff_width <- target_width - img_current_width
        # img_diff_height <- target_height - img_current_height
        # img_pad_width <- img_diff_width/2
        # img_pad_height <- img_diff_height/2
        # img <- image_border(img, "black", paste0(img_pad_width,"x",img_pad_height))
        
        img <- image_scale(img, paste0(target_width, "x", target_height, "!"))
        
        for(i in 1:length(mask_file[[1]])){
          
          single_mask <- image_read(mask_file[[1]][i])
          
          #---following not currently used - code for border
          # mask_current_width <-  unname(image_info(single_mask)[[2]])
          # mask_current_height <-  unname(image_info(single_mask)[[3]])
          # mask_diff_width <- target_width - mask_current_width
          # mask_diff_height <- target_height - mask_current_height
          # mask_pad_width <- mask_diff_width/2
          # mask_pad_height <- mask_diff_height/2
          # single_mask <- image_border(single_mask, "black", paste0(mask_pad_width,"x",mask_pad_height))
          
          #single_mask <- image_scale(single_mask, paste0(target_width, "x", target_height, "!"))
          
          if(i==1){
            mask <- single_mask
          }else{
            mask <- image_composite(mask, single_mask, operator = "Plus")
          }
        }
        
        return(list(img = img, mask = mask))
      }
      randomBSH <- function(img,
                            u = 0,
                            brightness_shift_lim = c(90, 110), # percentage
                            saturation_shift_lim = c(95, 105), # of current value
                            hue_shift_lim = c(80, 120)) {
        
        if (rnorm(1) < u) return(img)
        
        brightness_shift <- runif(1, 
                                  brightness_shift_lim[1], 
                                  brightness_shift_lim[2])
        saturation_shift <- runif(1, 
                                  saturation_shift_lim[1], 
                                  saturation_shift_lim[2])
        hue_shift <- runif(1, 
                           hue_shift_lim[1], 
                           hue_shift_lim[2])
        img <- image_modulate(img, 
                              brightness = brightness_shift, 
                              saturation =  saturation_shift, 
                              hue = hue_shift)
        img
      }
      randomHorizontalFlip <- function(img, 
                                       mask,
                                       u = 0) {
        if (rnorm(1) < u) return(list(img = img, mask = mask))
        list(img = image_flop(img), mask = image_flop(mask))
      }
      
      img2arr <- function(image, 
                          target_width = 256,
                          target_height = 256) {
        result <- aperm(as.numeric(image[[1]])[,, 1:3], c(2, 1, 3)) # transpose
        dim(result) <- c(1, target_width, target_height, 3)
        return(result)
      }
      
      mask2arr <- function(mask,
                           target_width = 256,
                           target_height = 256) {
        result <- t(as.numeric(mask[[1]])[,, 1]) # transpose
        dim(result) <- c(1, target_width, target_height, 1)
        return(result)
      }
    })
    
    registerDoParallel(cl)
    
    # start new epoch
    if (length(images_iter) < batch_size) {
      images_iter <<- images_all
      masks_iter <<- masks_all
    }
    
    batch_ind <- sample(1:length(images_iter), batch_size)
    batch_images_list <- images_iter[batch_ind]
    images_iter <<- images_iter[-batch_ind]
    batch_masks_list <- masks_iter[batch_ind]
    masks_iter <<- masks_iter[-batch_ind]
    
    x_y_batch <- foreach(i = 1:batch_size) %dopar% {
      x_y_imgs <- imagesRead(image_file = batch_images_list[i],
                             mask_file = batch_masks_list[i])
      # without augmentation
      # return as arrays
      x_y_arr <- list(x = img2arr(x_y_imgs$img),
                      y = mask2arr(x_y_imgs$mask))
    }
    
    stopCluster(cl)
    
    x_y_batch <- purrr::transpose(x_y_batch)
    x_batch <- do.call(abind, c(x_y_batch$x, list(along = 1)))
    y_batch <- do.call(abind, c(x_y_batch$y, list(along = 1)))
    result <- list(keras_array(x_batch), 
                   keras_array(y_batch))
    return(result)
  }
}