source('terugrekenfunctie.r')
library(tensorflow)
#selecteer het plaatje
i = 1


################################################Kijk naar connected layer achter output layer

#uitgaand signaal
signaal0 = sess$run(y_conv, feed_dict = dict(x = test, keep_prob=1.0))[i,]

#maak uitkomst binair
signaal0[signaal0 == max(signaal0)] = 1
signaal0[signaal0<1]=0




#welke gewichten heeft de output laag?
neuron_gewichten = sess$run(W_fc2,  feed_dict = dict(x = test, keep_prob=1.0))
#wat zijn de biases van de output laag?
neuron_bias = sess$run(b_fc2,  feed_dict = dict(x = test, keep_prob=1.0))
#wat is de input van de output layer
input = sess$run(h_fc1, feed_dict = dict(x = test, keep_prob=1.0))[i,]


signaal1 = terugreken(neuron_gewichten = neuron_gewichten, neuron_bias = neuron_bias , input = input, signaal1 = signaal0)


#welke gewichten heeft de output laag?
neuron_gewichten = sess$run(W_fc1,  feed_dict = dict(x = test, keep_prob=1.0))
#wat zijn de biases van de output laag?
neuron_bias = sess$run(b_fc1,  feed_dict = dict(x = test, keep_prob=1.0))
#wat is de input van de output layer
input = sess$run(h_conv3_flat, feed_dict = dict(x = test, keep_prob=1.0))[i,]




signaal2 = terugreken(neuron_gewichten = neuron_gewichten, neuron_bias = neuron_bias , input = input, signaal1 = signaal1)



####vorm terug naar plaatje



plaatje = array( signaal2, dim = c(64,15,15))
plaatje = aperm(plaatje, c(3,2,1))
plaatje = apply(plaatje, c(1,2), sum)

image(plaatje)

