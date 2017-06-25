#Deze functie is een iteratieve stap waarin wordt gekeken welke neuronen in welke mate bijdragen aan de eindconclusie van het neurale netwerk
#deze functie werkt voor fully connected layers

#neuron gewichten van de laag
#neuron biases van de laag
# input van de laag (output van de laag daarvoor
#resultaat van voorgaande iteratie

terugreken = function(neuron_gewichten, neuron_bias, input, signaal1 = signaal){
signalen = matrix(c(1:dim(neuron_gewichten)[2]), ncol = 1)

signalen = apply( signalen, 1, function(z){
  neuron = z[1]
  #het signaal dat het specifieke neuron heeft geactiveerd
  sign = as.vector(input) * as.vector(neuron_gewichten[,neuron]) + as.vector( neuron_bias[neuron] / length(neuron_bias) )
  return(sign)
})

signaal2 = apply(signalen, 1, function(z){
  return(z * signaal1)
  
})
signaal2 = colSums(signaal2)

return(signaal2)
}

#reken nieuw signaal uit door
#signaal = terugreken(neuron_gewichten = neuron_gewichten, neuron_bias = neuron_bias , input = input, signaal1 = signaal)








