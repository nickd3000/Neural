package com.physmo.neural;

import com.physmo.minvio.BasicDisplay;

import java.awt.*;

public class NN2Renderer {

    NN2 nn2 = null;
    BasicDisplay bd = null;
    Color colBg = new Color(43, 78, 135);
    Color colNode = new Color(251, 88, 91);
    int dx,dy,dw,dh;

    public NN2Renderer(NN2 nn2, BasicDisplay bd, int dx, int dy, int dw, int dh) {
        this.nn2 = nn2;
        this.bd = bd;
        this.dx=dx;
        this.dy=dy;
        this.dw=dw;
        this.dh=dh;
    }

    public void draw() {
        int numLayers = nn2.nodeLayers.size();

        bd.setDrawColor(colBg);
        bd.drawFilledRect(dx,dy,dw,dh);
        bd.setDrawColor(colNode);

        // Draw connections.
        for (int l=0;l<numLayers-1;l++) {
            int ncThis = nn2.nodeLayers.get(l).size;
            int ncNext = nn2.nodeLayers.get(l+1).size;
            for (int a=0;a<ncThis;a++) {
                for (int b=0;b<ncNext;b++) {
                    int [] posA = getPosition(l,a);
                    int [] posB = getPosition(l+1,b);
                    double v = nn2.weightLayers.get(l).weights[a];
                    drawConnection(dx+posA[0],dy+posA[1],dx+posB[0],dy+posB[1],v);
                }
            }

        }


        // Draw nodes.
        for (int l=0;l<numLayers;l++) {
            int [] pos = getPosition(l,0);

            for (int n=0;n<nn2.nodeLayers.get(l).size;n++) {
                pos = getPosition(l,n);
                drawNode(dx + pos[0], dy + pos[1], 1);
            }
        }
    }

    public void drawNode(int x, int y, double v) {
        double nodeSize = 5;
        bd.drawFilledCircle(x-nodeSize/2,y-nodeSize/2,nodeSize);
    }

    public void drawConnection(int x, int y, int x2, int y2,double v) {
        double nodeSize = 5;
        //bd.drawCircle(x-nodeSize/2,y-nodeSize/2,nodeSize);
        int iv = (int)((v+0.5)*128.0);
        if (iv<0) iv=0;
        if (iv>0xff) iv = 0xff;

        Color c = new Color(iv,iv,iv);
        bd.setDrawColor(c);

        bd.drawLine(x,y,x2,y2);
    }

    public int[] getPosition(int layer, int index) {
        int[] point = {0,0};
        int numLayers = nn2.nodeLayers.size();
        int layerX = 0;
        int padding = 10;
        int widthA = dw-(padding*2);
        int span = widthA / numLayers;
        int yspan = (dh-(padding*2))/nn2.nodeLayers.get(layer).size;
        layerX = (span*layer);
        point[0] = padding+layerX+(span/2);
        point[1] = padding+(yspan*index)+(yspan/2);
        return point;
    }

}
