package com.physmo.neural.old2;

class Node {
    double value;
    double sum;
    double error;
    int layerId;
    int nodeSequence; // Index of node in this layer.

    Node() {
        value = sum = error = 0;
    }
}
