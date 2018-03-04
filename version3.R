library (keras)
library (magick)
library (abind)
library (reticulate)
library (parallel)
library (doParallel)
library (foreach) 
library (microbenchmark)
library (spatstat)

#set working dir
setwd("~/Kaggle/2018 Data Science Bowl - Nuclei")

source(file = "train_generator.R")
source(file = "val_generator.R")

#------------------------------------------------------------------#
# Parameters
#------------------------------------------------------------------#

epochs <- 2 # The number of epochs
batch_size <- 5 # Batch size
#train_samples <- 670 # Size of training sample
train_samples <- 100 # Reduced size of training sample for testing purpose
train_index <- sample (1: train_samples, round (train_samples * 0.8)) # 80%
val_index <- c (1: train_samples) [- train_index]

#------------------------------------------------------------------#
# get path names for images
#------------------------------------------------------------------#

TRAIN_PATH =paste0(getwd(), '/stage1_train/', sep = "") 
TEST_PATH = paste0(getwd(), '/stage1_test/', sep = "")

#get train and test id's 
train_ids <- dir(path = TRAIN_PATH) 
test_ids <- dir(path = TEST_PATH)

image_paths_train <- character()
image_paths_test <- character()

#preprocess images, ensure they are resized.
for (n in 1:length(train_ids)){
  
  id_ <- train_ids[n]
  path <- paste0(TRAIN_PATH, id_)
  image_paths_train[n] = paste0(path, '/images/', id_, '.png')
}

for (n in 1:length(test_ids)){
  
  id_ <- test_ids[n]
  path <- paste0(TEST_PATH, id_)
  image_paths_test[n] = paste0(path, '/images/', id_, '.png')
}

#------------------------------------------------------------------#
# get path names for masks
#------------------------------------------------------------------#

# store mask paths in list since there are unequal numbers for each image

mask_paths <- list()

for (n in 1:length(train_ids)){
  
  id_ <- train_ids[n]
  path <- paste0(TRAIN_PATH, id_)
  
  # open and, aggregate and resize masks
  mask_ids <- dir(path = paste0(path, '/masks/' ))
  path_vec <- character()
  
  for(k in 1:length(mask_ids)){
    path_vec[k] <- paste0(path, '/masks/', mask_ids[k])
  }
  
  mask_paths[[n]] <- path_vec
}

#------------------------------------------------------------------#
# check dimensions of images 
#------------------------------------------------------------------#

lst_dim_summary <- list()
for (i in 1:length(image_paths_train)){
   im_ob <- image_read(image_paths_train[i])
  lst_dim_summary[[i]] <- data.frame(image_info(im_ob)[2], image_info(im_ob)[3])
}
dtf_dim_summary <- do.call("rbind", lst_dim_summary)
table(dtf_dim_summary)

#------------------------------------------------------------------#
# define iterators
#------------------------------------------------------------------#

train_iterator <- py_iterator(train_generator(image_paths = image_paths_train,
                                              mask_paths = mask_paths,
                                              samples_index = train_index,
                                              batch_size = batch_size))

val_iterator <- py_iterator(val_generator(image_paths = image_paths_train,
                                          mask_paths = mask_paths,
                                          samples_index = val_index,
                                          batch_size = batch_size))

#------------------------------------------------------------------#
# Segmentation and the Loss Function
#------------------------------------------------------------------#

K <- backend()
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- K$flatten(y_true)
  y_pred_f <- K$flatten(y_pred)
  intersection <- K$sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / 
    (K$sum(y_true_f) + K$sum(y_pred_f) + smooth)
  return(result)
}


bce_dice_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
}

#------------------------------------------------------------------#
# Model architecture 
#------------------------------------------------------------------#

get_unet_256 <- function(input_shape = c(256, 256, 3),
                         num_classes = 1) {
  
  inputs <- layer_input(shape = input_shape)

  down1 <- inputs %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
 
  down2 <- down1_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))

  down3 <- down2_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))

  center <- down3_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  
  up3 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")

  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")

  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
  

  classify <- layer_conv_2d(up1,
                            filters = num_classes, 
                            kernel_size = c(1, 1),
                            activation = "sigmoid")
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  model %>% compile(
    optimizer = optimizer_rmsprop(lr = 0.0001),
    loss = bce_dice_loss,
    metrics = c(dice_coef)
  )
  return(model)
}
model <- get_unet_256()


#------------------------------------------------------------------#
# Train model
#------------------------------------------------------------------#

tensorboard("logs_r")

callbacks_list <- list(
  callback_tensorboard("logs_r"),
  callback_early_stopping(monitor = "val_python_function",
                          min_delta = 1e-3,
                          patience = 6,
                          verbose = 1,
                          mode = "max"),
  callback_reduce_lr_on_plateau(monitor = "val_python_function",
                                factor = 0.1,
                                patience = 4,
                                verbose = 1,
                                epsilon = 1e-3,
                                mode = "max"),
  callback_model_checkpoint(filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                            monitor = "val_python_function",
                            save_best_only = TRUE,
                            save_weights_only = TRUE,
                            mode = "max" )
)

model %>% fit_generator(
  train_iterator,
  steps_per_epoch = as.integer(length(train_index) / batch_size),
  epochs = epochs,
  validation_data = val_iterator,
  validation_steps = as.integer(length(val_index) / batch_size),
  verbose = 1,
  view_metrics = T,
  callbacks = callbacks_list
)

#save_model_hdf5(model, filepath = "C:\\Users\\apavlides\\Documents\\Kaggle\\2018 Data Science Bowl - Nuclei\\large_image.h5")

#------------------------------------------------------------------#
# Model predictions
#------------------------------------------------------------------#


test_samples <- 65
test_index <- sample(1:test_samples, 65) 
load_model_weights_hdf5(model, "weights.24-0.25.hdf5") # best model
pred_batch_size <- 5 # Batch size

imageRead <- function(image_file,
                      target_width = 256, 
                      target_height = 256) {
  img <- image_read(image_file)
  img <- image_scale(img, paste0(target_width, "x", target_height, "!"))
}

img2arr <- function(image, 
                    target_width = 256,
                    target_height = 256) {
  result <- aperm(as.numeric(image[[1]])[, , 1:3], c(2, 1, 3)) # transpose
  dim(result) <- c(1, target_width, target_height, 3)
  return(result)
}

arr2img <- function(arr,
                    target_width = 256,
                    target_height = 256) {
  img <- image_read(arr)
  img <- image_scale(img, paste0(target_width, "x", target_height, "!"))
}

qrle_vector <- function(mask) {
browser()
  img <- mask
  dim(img) <- c(256, 256, 1)
  img <- arr2img(img)
  arr <- as.numeric(img[[1]])[, , 1]

  rotate <- function(x) t(apply(x, 2, rev))
  arr <- rotate(arr) %>% rotate() %>% rotate() 
  vect <- as.vector(arr) 
  turnpoints <- c(vect, 0) - c(0, vect)  
  starts <- which(turnpoints == 1)  
  ends <- which(turnpoints == -1)  
  # 1 indexed 
  paste(c(rbind(starts + 1, ends - starts)), collapse = " ") 
}

qrle_2d_image <- function(mask, image_id) {

  mask <- ifelse(mask<.5,0,1)
  
  # kern = makeBrush(3, shape='line')
  # mask_ <- EBImage::opening(mask, kern = kern)
  
  arr <- EBImage::bwlabel(mask)
  
  # convert to im object (two-dimensional pixel image)
  arr <- as.im(arr)
  
  # split im object, we now have a mask for each blob in image
  split_arr <- spatstat::split.im(x = arr, f = arr)
  
  # remove first element, which shows backgroud
  split_arr[1] <- NULL
  
  # iterate through each mask and get its run-length encoding
  result <- list()
  for (i in 1:length(split_arr)){
    
    current_img <- split_arr[[i]]
    y_ <- as.matrix.im(current_img)
    y_ <- ifelse(is.na(y_),0,1)
    y <- qrle_vector(y_)
    result[[i]] <- data.frame("ImageId" = image_id, "EncodedPixels" = y)
    
  }
  do.call("rbind", result)
}

test_generator <- function(images_dir, 
                           samples_index,
                           pred_batch_size) {

  images_iter <- images_dir[samples_index] # for current epoch
  x_batch <- list()
  
  function() {

    if(length(images_iter) > 0){
      
      batch_images_list <- images_iter[1:pred_batch_size]
      images_iter <<- images_iter[-c(1:pred_batch_size)]
      
    for (i in 1:pred_batch_size) {
      img <- imageRead(image_file = batch_images_list[i])
      x_batch[[i]] <- img2arr(img)
      }
      #modify to return an array, not a list of arrays. 
      x_batch <- do.call(what = abind, args = c(x_batch, list(along = 1))) 
      result <- list(keras_array(x_batch))
      return(result)
      
      } else{
      
      return(NULL)
    }
  }
  
}

test_iterator <- py_iterator(test_generator(images_dir = image_paths_test,
                                            samples_index = test_index,
                                            pred_batch_size = pred_batch_size), completed = NULL)

# makes predictions on batches generated from the test_iterator. The test_iterator returns all 
# images in batches of pred_batch_size. If steps > 1 then it will repeat the predictions. Hense, 
# steps = 1. 
preds <- predict_generator(model, test_iterator, steps = 13)


# check images match with index 
image(img2arr(imageRead(image_paths_test[test_index[3]]))[,,,1])
image(preds[3,,,])

# add names to output
# get imageId of each image processed (in correct order)
img_names <- sapply(image_paths_test[test_index], FUN = function(x){strsplit(x, split = "/")[[1]][8]}) %>% unname()

# output using format specifed in kaggle
result <- list()
for(i in 1:65){
  result[[i]] <- qrle_2d_image(preds[i, , , ], img_names[i])
}
out <- do.call("rbind", result)


write.table(out,  file = "C:/Users/apavlides/Documents/Kaggle/2018 Data Science Bowl - Nuclei/40hourRun.csv",
            row.names=FALSE, sep=",")





