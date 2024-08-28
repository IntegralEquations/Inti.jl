SetFactory("OpenCASCADE");

Disk(1) = {0,0,0,1,0.5};
Disk(2) = {0,0,0,1.3,0.6};
Rectangle(3)={Cos(7*Pi/10),-5,0,-5,10};
BooleanDifference{Surface{2};Delete;}{Surface{1};Delete;}
BooleanDifference{Surface{2};Delete;}{Surface{3};Delete;}


