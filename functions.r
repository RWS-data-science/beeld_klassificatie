load_data = function(data_dir){
    dirs = list.dirs(path = data_dir, full.names = TRUE, recursive = FALSE);
    labels = c();
    images = c();
    for(dir in dirs){
        files = list.files(path = dir);
        for(file in files){
            labels <- as.integer(dir);
	    image <- readPNG(paste(dir, file, sep ='/'), FALSE);
            images <- image_to_array(image, 100, 100, 3);
        }
        
    }
    data= data.frame(cbind(labels, images))
    colnames(data) <- c("labels", "images")
    return(data);
}

image_to_array = function(image, w,h,c){
    #resize
print(dim(image));
    image <- resizeImage(image, w, h)
    print(dim(image));
    #flip the channel order
    image_data = aperm(image, c(3,2,1));
    
    #flatten
    image_data = matrix( image_data , ncol =1);
    #normalize
    image_data = normalize_image(image_data);
    
    #superfluous?
    #image_data = matrix(image_data, nrow = 1)
    return(image_data);
    #df = rbind(df, image_data)
}

array_to_image = function(array,w,h,c){
    return(Image(aperm(pl,array( image_data , dim = c(c,w,h)),  c(3,2,1))));
}

normalize_image = function(image_data){
    image_data<- scale( image_data,center = min(image_data) , scale = max(image_data)-min(image_data));
}

rotate3d = function(image_data,w,h,c){
  
    #alloceer ruimte in array
    a= array(, dim = c(w,h,c))
    #array to image
    image = array_to_image(image_data, w,h,c);
    
    #define a transformation matrix and apply
    x = sample(c(4:10), 1)/10
    y = sample(c(4:10), 1)/10
    
    m <- matrix(c(x, 0,
                0, y,
                0, 0), byrow = TRUE,nrow=3)
  
    img_affine = affine(image, m)
    img_affine = img_affine[0:(x*h) ,0: (y*w),]
  
    return(normalize_image(image_to_array(image)));
}
           
#functie voor het maken van gewichten
weight_variable <- function(shape,name) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  return(tf$Variable(initial,name))
}

#functie voor het maken van biasses
bias_variable <- function(shape, name) {
  initial <- tf$constant(0.1, shape=shape)
  return(tf$Variable(initial, name))
}

#Convolutie functie
conv2d <- function(x, W) {
  return(tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME'))
}
#poolfunctie
max_pool_2x2 <- function(x) {
  return( tf$nn$max_pool(x, ksize=c(1L, 2L, 2L, 1L),strides=c(1L, 2L, 2L, 1L), padding='SAME'))
}
           
save_network <- function() {
    saveRDS(sess$run(W_conv1), file = 'db/neuraalnet/w_conv1.rds')
    saveRDS( sess$run(b_conv1), file = 'db/neuraalnet/b_conv1.rds' )
    saveRDS(sess$run(W_conv2), file = 'db/neuraalnet/w_conv2.rds')
    saveRDS(sess$run(b_conv2), file = 'db/neuraalnet/b_conv2.rds')
    saveRDS(sess$run(W_fc1) , file = 'db/neuraalnet/w_fc1.rds')
    saveRDS(sess$run(b_fc1), file = 'db/neuraalnet/b_fc1.rds')
    saveRDS(sess$run(W_fc2), file =  'db/neuraalnet/w_fc2.rds' )
    saveRDS(sess$run(b_fc2), file =  'db/neuraalnet/b_fc2.rds' )
    saveRDS(sess$run(W_conv3), file =  'db/neuraalnet/w_conv3.rds' )
    saveRDS(sess$run(b_conv3), file =  'db/neuraalnet/b_conv3.rds' )
}

resizeImage = function(im, w.out, h.out) {
  # function to resize an image 
  # im = input image, w.out = target width, h.out = target height
  # Bonus: this works with non-square image scaling.
  
  # initial width/height
  w.in = nrow(im)
  h.in = ncol(im)
  
  # Create empty matrix
  im.out = matrix(rep(0,w.out*h.out), nrow =w.out, ncol=h.out )
  
  # Compute ratios -- final number of indices is n.out, spaced over range of 1:n.in
  w_ratio = w.in/w.out
  h_ratio = h.in/h.out
  
  # Do resizing -- select appropriate indices
  im.out <- im[ floor(w_ratio* 1:w.out), floor(h_ratio* 1:h.out), 3]
  
  return(im.out)
}


