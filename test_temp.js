var convnetjs = require("./convnet-min.js");

const mean = (arr)=>{
    return arr.reduce((acc, val) => acc + val, 0) / arr.length;
}

const standardDeviation = (arr, usePopulation = false) => {
    const meanArr = mean(arr);
    return Math.sqrt(
        arr
        .reduce((acc, val) => acc.concat((val - meanArr) ** 2), [])
        .reduce((acc, val) => acc + val, 0) /
        (arr.length - (usePopulation ? 0 : 1))
    );
};

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

let data = [
    38,
    39,
    41,
    42,
    45,
    47,
    49,
    51,
    53,
    55,
    58,
    60,
    62,
    66,
    69,
    71,
    74,
    77,
    80,
    83,
    87,
    90,
    94,
    98,
    101,
    113,
    120,
    126,
    131,
    136,
    140,
    145,
    149,
    153,
    158,
    161,
    166,
    170,
    173,
    178,
    181,
    185,
    189,
    193,
    200,
    208,
    214,
    219,
    225,
    229,
    233,
    237,
    242,
    257,
    262,
    267,
    272,
    277,
    281,
    285,
    291,
    297,
    301,
    305,
    309,
    315,
    319,
    321,
    324,
];

dataMean = mean(data);
dataStdDev = standardDeviation(data);
dataNorm = [];
for(var i in data) {
    dataNorm.push((data[i] - dataMean)/dataStdDev);
}

let labels = [
    75,
    74,
    73,
    72,
    71,
    70,
    69,
    68,
    67,
    66,
    65,
    64,
    63,
    62,
    61,
    60,
    59,
    58,
    57,
    56,
    55,
    54,
    53,
    52,
    51,
    50,
    49,
    48,
    47,
    46,
    45,
    44,
    43,
    42,
    41,
    40,
    39,
    38,
    37,
    36,
    35,
    34,
    33,
    32,
    31,
    30,
    29,
    28,
    27,
    26,
    25,
    24,
    23,
    20,
    19,
    18,
    17,
    16,
    15,
    14,
    13,
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
];

labelsNorm = [];
for(var i in labels) {
    labelsNorm.push(labels[i] / 100.0);
}
N = data.length;

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
for (var ep = 0; ep < 10; ep++) {
    avloss = 0.0;
    for (var iters = 0; iters < 50; iters++) {
        for (var ix = 0; ix < N; ix++) {
            netx.w[0] = dataNorm[ix];
            var stats = trainer.train(netx, [labelsNorm[ix]]);
            avloss += stats.loss;
        }
    }
    avloss /= N * iters;
    console.log(avloss);
}

// evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
// actual output by looking into its 'w' field:
var netx1 = new convnetjs.Vol(1, 1, 1);
netx1.w[0] = (265.0 - dataMean) / dataStdDev;
var predicted_values = net.forward(netx1);
console.log('predicted value: ' + predicted_values.w[0] * 100.0);