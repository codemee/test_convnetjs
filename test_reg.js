var convnetjs = require("./convnet-min.js");

var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: 1
});
layer_defs.push({
    type: 'fc',
    num_neurons: 20,
    activation: 'relu'
});
layer_defs.push({
    type: 'fc',
    num_neurons: 20,
    activation: 'sigmoid'
});
layer_defs.push({
    type: 'regression',
    num_neurons: 1
});
var net = new convnetjs.Net();
net.makeLayers(layer_defs);

N = 40
data = [];
labels = [];
for (var i = 0; i < N; i++) {
    var x = Math.random() * 10 - 5;
    // var y = s * Math.sin(x);
    // var y = 2 * x + 1;
    var y = x * x;
    data.push([x]);
    labels.push([y]);
}

// train on this datapoint, saying [0.5, -1.3] should map to value 0.7:
// note that in this case we are passing it a list, because in general
// we may want to  regress multiple outputs and in this special case we 
// used num_neurons:1 for the regression to only regress one.
var trainer = new convnetjs.SGDTrainer(net, {
    learning_rate: 0.01,
    momentum: 0.0,
    batch_size: 1,
    l2_decay: 0.001
});

var netx = new convnetjs.Vol(1, 1, 1);
for(var ep = 0; ep < 100; ep++){
    avloss = 0.0;
    for (var iters = 0; iters < 50; iters++) {
        for (var ix = 0; ix < N; ix++) {
            netx.w[0] = data[ix][0];
            var stats = trainer.train(netx, labels[ix]);
            avloss += stats.loss;
        }
    }
    avloss /= N * iters;
    console.log(avloss);
}

// evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
// actual output by looking into its 'w' field:
var netx1 = new convnetjs.Vol(1, 1, 1);
netx1.w[0] = 4;
var predicted_values = net.forward(netx1);
console.log('predicted value: ' + predicted_values.w[0]);