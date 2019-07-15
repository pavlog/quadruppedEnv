import math
import random

NUM_VERTS_X = 80
NUM_VERTS_Y = 80 #26
X_OFFSET = 3.0
Y_OFFSET = 0.0
totalVerts = NUM_VERTS_X*NUM_VERTS_Y
totalTriangles = 2*(NUM_VERTS_X-1)*(NUM_VERTS_Y-1)
offset = 0.0
TRIANGLE_SIZE = 0.2
waveheight=0.07
gGroundVertices = [None] * totalVerts*3
gGroundUVVertices = [None] * totalVerts*2
gGroundIndices = [None] * totalTriangles*3

i=0

for i in range (NUM_VERTS_X):		
	for j in range (NUM_VERTS_Y):
		x = (i-NUM_VERTS_X*0.5)*TRIANGLE_SIZE+X_OFFSET
		y = (j-NUM_VERTS_Y*0.5)*TRIANGLE_SIZE+Y_OFFSET
		gGroundVertices[(i+j*NUM_VERTS_X)*3+0] = x
		gGroundVertices[(i+j*NUM_VERTS_X)*3+1] = y
		if x>=-0.5 and x<=1.0 and y>=-0.5 and y<=0.5:
			gGroundVertices[(i+j*NUM_VERTS_X)*3+2] = 0.0
		else:
			#gGroundVertices[(i+j*NUM_VERTS_X)*3+2] = waveheight*math.sin(float(i))*math.cos(float(j)+offset)+random.random()*random.random()*waveheight
			gGroundVertices[(i+j*NUM_VERTS_X)*3+2] = random.random()*waveheight

		gGroundUVVertices[(i+j*NUM_VERTS_X)*2+0] = (i-NUM_VERTS_X*0.5)*TRIANGLE_SIZE
		gGroundUVVertices[(i+j*NUM_VERTS_X)*2+1] = (j-NUM_VERTS_Y*0.5)*TRIANGLE_SIZE


index=0
for i in range (NUM_VERTS_X-1):
	for j in range (NUM_VERTS_Y-1):
		gGroundIndices[index] = 1+j*NUM_VERTS_X+i
		index+=1
		gGroundIndices[index] = 1+j*NUM_VERTS_X+i+1
		index+=1
		gGroundIndices[index] = 1+(j+1)*NUM_VERTS_X+i+1
		index+=1
		gGroundIndices[index] = 1+j*NUM_VERTS_X+i
		index+=1
		gGroundIndices[index] = 1+(j+1)*NUM_VERTS_X+i+1
		index+=1
		gGroundIndices[index] = 1+(j+1)*NUM_VERTS_X+i
		index+=1

#print(gGroundVertices)
#print(gGroundIndices)

print("mtllib terrain.mtl")
print("o Terrain")

for i in range (totalVerts):
	print("v ",gGroundVertices[i*3+0],gGroundVertices[i*3+1],gGroundVertices[i*3+2])
	print("vt ",gGroundUVVertices[i*2+0],gGroundUVVertices[i*2+1])

print ("s on")
print ("usemtl Material")

for i in range (totalTriangles):
	print("f {}/{} {}/{} {}/{}".format(gGroundIndices[i*3+0],gGroundIndices[i*3+0],gGroundIndices[i*3+1],gGroundIndices[i*3+1],gGroundIndices[i*3+2],gGroundIndices[i*3+2]))
	

