
selectie = function(path, spl){
  klassen = list.files(path)
  
  
  selectie_train = data.frame()
  selectie_test = data.frame()
  
  
  for(i in 1:length(klassen)){
    files = paste0( path, '/', klassen[i], '/',  list.files(paste0(path, '/',klassen[i])))
    extra_train = sample(files, round(spl*length(files)))
    
    selectie_train = rbind(selectie_train, cbind( extra_train,  i))
    selectie_test = rbind( selectie_test ,cbind(setdiff(files, extra_train),i))
    
    
  }
  
  colnames(selectie_test) = c('files', 'labels' )
  selectie_test$files = as.character(selectie_test$files)
  selectie_test$labels = as.numeric(selectie_test$labels)
  
  
  colnames(selectie_train) = c('files', 'labels' )
  selectie_train$files = as.character(selectie_train$files)
  selectie_train$labels = as.numeric(selectie_train$labels)
  
  selectie = list(selectie_train, selectie_test)
  
  saveRDS(selectie, 'db/netwerk/selectie.rds')
  
  return(selectie)
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
  return( tf$nn$max_pool(x,     ksize=c(1L, 2L, 2L, 1L),strides=c(1L, 2L, 2L, 1L), padding='SAME'))
}


###lees beelden in
lees_in = function(files, w, h,  schaal){
  
  
  df = lapply(files, function(file){
   
    
    
    
    #lees foto in
    im <- readImage( file )
    
    im = noise(im = im)
    
    #resize
    im =  resize(im, w = w, h = h)
    #im <- channel(im, "grey")
    
    a = array(dim = c(w,h,3))
    a[,,1] = imageData(im[,,1])
    a[,,2] = imageData(im[,,2])
    a[,,3] = imageData(im[,,3])
    
  
    a = aperm(a, c(3,2,1))
    
    
    extra = matrix( a , ncol =1)
    
    #normaliseer df[N,,,]
    ma = max(extra)
    mi = min(extra)
    extra<- schaal* scale( extra,center = mi , scale = ma-mi)
    
    return(matrix(extra, nrow = 1))
    
  })
  
  df = do.call(rbind,df)
  
  return(df)
  
}

##############one hot
one_hot = function(labels){
   mat_hot = lapply(labels, function(label){
    hot =  matrix(c(0), nrow = 1, ncol =clas)
     hot[,label] = 1
     return(hot)
     
   })
  
   mat_hot = do.call(rbind, mat_hot)
   return(mat_hot)
}


#################ruis functie
noise = function(im){
  
im = im  + rnorm(im, mean = 0, sd = 0.05) 
  


x = sample(c(4:10), 1)/10
y = sample(c(4:10), 1)/10

m <- matrix(c(x, 0,
              0, y,
              0, 0), byrow = TRUE,nrow=3)


img_affine = affine(im, m)



img_affine = img_affine[0:(x*dim(im)[1]) ,0: (y*dim(im)[2]),]



  
  return(img_affine)
}
