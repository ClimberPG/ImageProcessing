function display_faces(faces, numPerLine, ShowLine)

faceW = 32;
faceH = 32;

if nargin == 1
    numPerLine = 16;
    ShowLine = 2;
end

Y = zeros(faceH*ShowLine,faceW*numPerLine);
for i=0:ShowLine-1
   for j=0:numPerLine-1
     curr_face = reshape(faces(i*numPerLine+j+1,:),[faceW faceH]);
     Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = curr_face;
   end
end

%imagesc(Y);colormap(gray);

imshow(Y,[]);