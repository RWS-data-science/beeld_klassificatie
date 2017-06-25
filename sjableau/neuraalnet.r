############
#library
library(tensorflow)
library(EBImage)
source('functies.r')
################

#####
#maak mappen
if(!dir.exists('db/netwerk')){
  dir.create('db/netwerk')
}
###################


#############
#zet parametrs
selectie =  selectie(spl = 0.8, path = "/home/beheerder/R/beeldenbank/gelabeled/borden_closeup/plaatjes")
selectie_train = selectie[[1]]
selectie_test = selectie[[2]]
aantal_pool  = 3
aantal_kanalen = 64

clas = 62#aantal klassen
schaal = 250 #pixel waarden tussen 0 en schaal
max_accuracy = 0.97 #opslaan vanaf

out = 0.5 #dropout
batch_train = 50 #batchsize
batch_test = 100 #batchsize test
ds = 0.999 #gradient descent
lr = 1e-3 #learningrate

h = as.integer(224) #heigth image
w = as.integer(224) #width image
kanalen = as.integer(3) #kanalen image
###############


#############
#place holders
#input
x <- tf$placeholder(tf$float32, shape(NULL, h*w*kanalen))
#target values
labels <- tf$placeholder(tf$float32, shape(NULL,clas))
#dropout rate
keep_prob <- tf$placeholder(tf$float32)
#learningrate
lrate <- tf$placeholder(tf$float32)
##################



##############
#laden van pre-trained layers
graph =  tf$train$NewCheckpointReader("db/vgg_16.ckpt")
lijst = graph$get_variable_to_shape_map()
############


###############
#declaraties van variabelen en constanten
#input
#reshape
#laag 1
b_conv1_1 =  graph$get_tensor("vgg_16/conv1/conv1_1/biases")
w_conv1_1 =  graph$get_tensor("vgg_16/conv1/conv1_1/weights")
#laag 2
b_conv1_2 =  graph$get_tensor("vgg_16/conv1/conv1_2/biases")
w_conv1_2 =  graph$get_tensor("vgg_16/conv1/conv1_2/weights")
#pool
#laag 3
w_conv2 <- weight_variable(shape(5L, 5L, 64, 40L), 'w_conv2')
b_conv2 <- bias_variable(shape(40L), 'b_conv2')
#pool
#laag 4
w_conv3 <- weight_variable(shape = shape(5L, 5L, 40L, 64L), 'w_conv3')
b_conv3 <- bias_variable(shape = shape(64L), 'b_conv3')
#pool
#laag 5
w_conv4 <- weight_variable(shape = shape(5L, 5L, 64L, 64L), 'w_conv4')
b_conv4 <- bias_variable(shape = shape(64L), 'b_conv4')
#reshape
#laag 6
w_fc1 <- weight_variable(shape((w*h)/(4^(aantal_pool)) * aantal_kanalen, 1024L), 'w_fc1')
b_fc1 <- bias_variable(shape(1024L), 'b_fc1')
#dropout
#laag 7
w_output <- weight_variable(shape(1024L, clas), 'w_output')
b_output <- bias_variable(shape(clas), 'b_output')
######################







###################### De graph
#input reshape
h_input = tf$reshape(x, shape(-1L,h, w, kanalen))
#convolutie laag 1_1
h_conv1_1 <- tf$nn$relu(conv2d(h_input, w_conv1_1) + b_conv1_1)
#convolutie laag 1_2
h_conv1_2 <- tf$nn$relu(conv2d(h_conv1_1, w_conv1_2) + b_conv1_2)
#poollaag 1
h_pool1 <- max_pool_2x2(h_conv1_2)
#convolutie laag 2
h_conv2 <- tf$nn$relu(conv2d(h_pool1, w_conv2) + b_conv2)
#poollaag 2
h_pool2 <- max_pool_2x2(h_conv2)
#convolutielaag 3
h_conv3 <- tf$nn$relu(conv2d(h_pool2, w_conv3) + b_conv3)
#poollaag 3
h_pool3 <- max_pool_2x2(h_conv3)
#convolutielaag 4
h_conv4 <- tf$nn$relu(conv2d(h_pool3, w_conv4) + b_conv4)
#reshape
h_conv4_flat <- tf$reshape(h_conv4, shape(-1L, (w*h)/(4^(aantal_pool)) * aantal_kanalen))
#connectedlaag 1
h_fc1 <- tf$nn$relu(tf$matmul(h_conv4_flat, w_fc1) + b_fc1)
#drop out
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)
#output
h_output <- tf$nn$softmax(tf$matmul(h_fc1_drop, w_output) + b_output)
###############



################
#fout functie
cross_entropy = tf$reduce_mean(-tf$reduce_sum(labels * tf$log( tf$clip_by_value(h_output, 1e-10,1 )), reduction_indices=1L))
#trainstep met adam optimizer (met momentum)
train_step <- tf$train$AdamOptimizer(lrate)$minimize(cross_entropy)
#bereken percentage goede antwoorden
correct_prediction <- tf$equal(tf$argmax(h_output, 1L), tf$argmax(labels, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
#####################


###########
#maak sessie
sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())
#run sessie
#output = sess$run(h_output, feed_dict = dict(x = train , labels = train_labels , keep_prob = out, lrate = ds^i*lr))
###########



####################
#train
for (i in 1:20000) {
  
  #lees 50 random plaatjes in
  samp = sample( x=  c(1: nrow(selectie_train)) , size = batch_train )
  
  train_labels =one_hot(selectie_train$labels[samp])
  
  files = selectie_train$files[samp]
  train = lees_in (files=files ,w=w,h=h, schaal = schaal)
  
  
  #train met gradient descent
  train_step$run(feed_dict = dict(x = train , labels = train_labels , keep_prob = out, lrate = ds^i*lr))
  
  
  
  
  
  #valideer om de 100 keer hoe het gaat op de testset
  if (i %% 100 == 0) {
    #evalueer op de testset
    samp = sample( x=  c(1: nrow(selectie_test)) , size = batch_test )
    test_labels =one_hot(selectie_test$labels[samp])
    test = lees_in (files=selectie_test$files[samp] ,w=w,h=h, schaal = schaal)
    test_accuracy <- accuracy$eval(feed_dict = dict(  x = test, labels = test_labels, keep_prob = 1.0))
    print( paste("step:", i, "test accuracy:", test_accuracy) )
    
    #sla op als performance op testset verbeterd
    if(test_accuracy > max_accuracy  ){

      
      
      saveRDS( graph$get_tensor("vgg_16/conv1/conv1_1/weights")  , file = 'db/netwerk/w_conv1_1.rds')
      saveRDS(   graph$get_tensor("vgg_16/conv1/conv1_1/biases")  , file = 'db/netwerk/b_conv1_1.rds' )
      saveRDS(   graph$get_tensor("vgg_16/conv1/conv1_2/weights")  , file = 'db/netwerk/w_conv1_2.rds')
      saveRDS(   graph$get_tensor("vgg_16/conv1/conv1_2/biases")  , file = 'db/netwerk/b_conv1_2.rds' )
      saveRDS(sess$run(w_conv2), file = 'db/netwerk/w_conv2.rds')
      saveRDS( sess$run(b_conv2), file = 'db/netwerk/b_conv2.rds' )
      saveRDS(sess$run(w_conv3), file = 'db/netwerk/w_conv3.rds')
      saveRDS(sess$run(b_conv3), file = 'db/netwerk/b_conv3.rds')
      saveRDS(sess$run(w_conv4), file =  'db/netwerk/w_conv4.rds' )
      saveRDS(sess$run(b_conv4), file =  'db/netwerk/b_conv4.rds' )
      saveRDS(sess$run(w_fc1) , file = 'db/netwerk/w_fc1.rds')
      saveRDS(sess$run(b_fc1), file = 'db/netwerk/b_fc1.rds')
      saveRDS(sess$run(w_output), file =  'db/netwerk/w_output.rds' )
      saveRDS(sess$run(b_output), file =  'db/netwerk/b_output.rds' )
      
      max_accuracy = test_accuracy
      print('nieuw reccord')
    }
    
 
    
  }
  
  

  
  
  

}




##############################3
#laad netwerk weer in

#saver$restore(sess, "model.ckpt")
# 
# w_conv1_1= readRDS( file = 'db/netwerk/w_conv1_1.rds')
# w_conv1_1 = tf$cast(w_conv1_1, tf$float32)
# 
# b_conv1_1 = readRDS(  file = 'db/netwerk/b_conv1_1.rds' )
# b_conv1_1 = tf$cast(b_conv1_1, tf$float32)
# 
# w_conv1_2 = readRDS( file = 'db/netwerk/w_conv1_2.rds')
# w_conv1_2 = tf$cast(w_conv1_2, tf$float32)
# 
# b_conv1_2 = readRDS(  file = 'db/netwerk/b_conv1_2.rds' )
# b_conv1_2 = tf$cast(b_conv1_2, tf$float32)
# 
# w_conv2 = readRDS( file = 'db/netwerk/w_conv2.rds')
# w_conv2 = tf$cast(w_conv2, tf$float32)
# 
# b_conv2 = readRDS(  file = 'db/netwerk/b_conv2.rds' )
# b_conv2 = tf$cast(b_conv2, tf$float32)
# 
# w_conv3= readRDS( file = 'db/netwerk/w_conv3.rds')
# w_conv3 = tf$cast(w_conv3, tf$float32)
# 
# b_conv3 = readRDS( file = 'db/netwerk/b_conv3.rds')
# b_conv3 = tf$cast(b_conv3, tf$float32)
# 
# 
# w_conv4 = readRDS( file =  'db/netwerk/w_conv4.rds' )
# w_conv4 = tf$cast(w_conv4, tf$float32)
# 
# b_conv4 = readRDS( file =  'db/netwerk/b_conv4.rds' )
# b_conv4 = tf$cast(b_conv4, tf$float32)
# 
# w_fc1 = readRDS( file = 'db/netwerk/w_fc1.rds')
# w_fc1 = tf$cast(w_fc1, tf$float32)
# 
# b_fc1 = readRDS( file = 'db/netwerk/b_fc1.rds')
# b_fc1 = tf$cast(b_fc1, tf$float32)
# 
# w_output = readRDS( file =  'db/netwerk/w_output.rds' )
# w_output = tf$cast(w_output, tf$float32)
# 
# b_output = readRDS( file =  'db/netwerk/b_output.rds' )
# b_output = tf$cast(b_output, tf$float32)
