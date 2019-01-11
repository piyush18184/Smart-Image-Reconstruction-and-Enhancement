clc;
clear all;
close all;
% Reading the image
a=1;
while a>0
disp(' ________________________________________________ ')
disp('|-------------------WELCOME----------------------|')
disp('|                  1. Underwater Image           |')
disp('|                  2. Foggy Road                 |')
disp('|                  3. Foggy Bridge               |')
disp('|                  4. Dark Office                |')
disp('|                  5. All                        |')
disp('|                                                |')
disp('|          Press Any Other Key to Exit           |')
disp('|________________________________________________|')
in=input('|-----------PLEASE ENTER THE RESPONSE------------|\n');
if in == 1
    img = imread('C:\Users\ayush\Desktop\Blurred Images\Underwater.jpg');
    ImgEnh(img)
elseif in == 2
    img = imread('C:\Users\ayush\Desktop\Blurred Images\foggyroad.jpg');
    ImgEnh(img)
elseif in == 3
    img = imread('C:\Users\ayush\Desktop\Blurred Images\foggysf2.jpg');
    ImgEnh(img)
elseif in == 4
    img = imread('C:\Users\ayush\Desktop\Blurred Images\office_1.jpg');
    ImgEnh(img)
elseif in == 5
    img = imread('C:\Users\ayush\Desktop\Blurred Images\Underwater.jpg');
    ImgEnh(img)
    img = imread('C:\Users\ayush\Desktop\Blurred Images\foggyroad.jpg');
    ImgEnh(img)
    img = imread('C:\Users\ayush\Desktop\Blurred Images\foggysf2.jpg');
    ImgEnh(img)
    img = imread('C:\Users\ayush\Desktop\Blurred Images\office_1.jpg');
    ImgEnh(img)
else
    disp('Please Wait...Exiting..')
    input('Press Any Key to Continue')
    a=0;
end
end
function ImgEnh(img)
    [m, n, p] = size(img);
    imRGB_orig = zeros(p, m * n);
    num = 255;
    % Converting P,M,N to P,M*N and equating the saturation level for the image
    % If image is M,N,P
    if ndims(img) == 3
    red=img(:,:,1);
    total_red=sum(sum(red));
    green=img(:,:,2);
    total_green=sum(sum(green));
    blue=img(:,:,3);
    total_blue=sum(sum(blue));
    if total_red>total_green
        if total_red>total_blue
            maximum=total_red;
        else
            maximum=total_blue;
        end
    elseif total_green>total_blue
        maximum=total_green;
    else
        maximum=total_blue;
    end
    total=[total_red total_green total_blue];
    a=zeros(1:3);
    b=zeros(1:3);
    % Setting up the saturation level for M,N,P
    for i=1:3
        a(i) = 0.005*maximum/total(i);
        b(i) = 0.005*maximum/total(i);
    end
    for i = 1:p
    imRGB_orig(i,:) = reshape(double(img(:,:,i)),[1,m*n]);
    end
    % If image is M*N only
    else
    % Setting up the saturation level for M,N
    a = 0.001;
    b = 0.005;
    [m,n] = size(img);
    p = 1;
    imRGB_orig = reshape(double(img),[1,m*n]);
    end

    % Normalization technique
    RGBnorm = zeros(size(imRGB_orig));
    for ch = 1:p
    % Normalization done in order bring all the R,G & B contents to similar
    % level ie leveling up the minimum values to a median value of these R,G,B
    q = [a(ch),1-b(ch)];
    median = quantile(imRGB_orig(ch,:),q);
    temp = imRGB_orig(ch,:);
    temp(find(temp<median(1)))=median(1);
    temp(find(temp>median(2)))=median(2);
    RGBnorm(ch,:) = temp;
    bottom = min(RGBnorm(ch,:)); 
    top = max(RGBnorm(ch,:));
    % After normalizing the values, spreading them across the [0,255] range
    RGBnorm(ch,:) = (RGBnorm(ch,:)-bottom)*num/(top-bottom); 
    end


    if ndims(img) == 3
    outval = zeros(size(img));
    % Converting back from P,M*N to P,M,N
    for i = 1:p
    outval(:,:,i) = reshape(RGBnorm(i,:),[m, n]); 
    end
    else
    outval = reshape(RGBnorm,[m,n]); 
    end
    % outval here stores the normalized image
    outval = uint8(outval);

    % Converting the normalized image from RGB to LAB form
    img1=outval;
    cform = makecform('srgb2lab');
    lab1 = applycform(img1,cform);
    % Adaptive Equalization technique employed to futher enhance the normalized
    % image
    lab2 = lab1;
    lab2(:,:,1) = adapthisteq(lab2(:,:,1));
    cform = makecform('lab2srgb');
    % img2 stores the equalized image
    img2 = uint8(applycform(lab2,cform));


    % Image Sharpening on Lab components
    % lab1 holds normalized components(contrast stretching) and 
    % lab2 holds the equalized components (get uniform shape across the range)
    R1 = double(lab1(:,:,1))/255;
    R2 = double(lab2(:,:,1))/255;
    % To eliminate the zero-padding effects around the edge of the image,imfilter 
    % provides alternative boundary padding method called border replication.
    % In border replication, the value of any pixel outside the image is 
    % determined by replicating the value from the nearest border pixel.
    % WL1 & WL2 calculation
    % Laplacian highlights rapid intensity change(edge detection)
    WL1 = abs(imfilter(R1,fspecial('Laplacian')));
    WL2 = abs(imfilter(R2,fspecial('Laplacian')));
    % WC1 & WC2 calculation
    % Gaussian to counter the effect of noise sensitiviy
    h = [1/145 11/145 121/145 11/145 1/145];
    WC1 = imfilter(R1,h'*h);
    WC2 = imfilter(R2,h'*h);
    % LoG operation carried out to get the sharpened image
    WC1 = (WL1-WC1).^2;
    WC2 = (WL2-WC2).^2;
    % WS1 calculation by working on the Lab components of the 
    % LPF of normalized image above(i.e. outval)
    gfrgb = imfilter(img1,fspecial('gaussian'));
    cform = makecform('srgb2lab');
    lab = applycform(gfrgb,cform);
    l = double(lab(:,:,1)); 
    lm = mean(mean(l));
    a = double(lab(:,:,2)); 
    am = mean(mean(a));
    b = double(lab(:,:,3)); 
    bm = mean(mean(b));
    Diff_l=(l-lm).^2;
    Diff_a=(a-am).^2;
    Diff_b=(b-bm).^2;
    WS1 = Diff_l+Diff_a+Diff_b;
    gfrgb = imfilter(img2, fspecial('gaussian'));
    lab = applycform(gfrgb,cform);
    l = double(lab(:,:,1)); 
    lm = mean(mean(l));
    a = double(lab(:,:,2)); 
    am = mean(mean(a));
    b = double(lab(:,:,3)); 
    bm = mean(mean(b));
    Diff_l=(l-lm).^2;
    Diff_a=(a-am).^2;
    Diff_b=(b-bm).^2;
    WS2 = Diff_l+Diff_a+Diff_b;
    % WE1 calculation by using the original gaussian equation
    sigma = 0.25;
    mu = 0.5;
    WE1 = (1/(sqrt(2*pi)*sigma))*exp(-(R1-mu).^2/(2*sigma^2));
    WE2 = (1/(sqrt(2*pi)*sigma))*exp(-(R2-mu).^2/(2*sigma^2));
    % W1 and W2 are the average weights to be used for Multi Scale Fusion Process 
    W1 = (WL1+WC1+WS1+WE1)./(WL1+WC1+WS1+WE1+WL2+WC2+WS2+WE2);
    W2 = (WL2+WC2+WS2+WE2)./(WL1+WC1+WS1+WE1+WL2+WC2+WS2+WE2);


    Wt1{1} = imfilter(W1, h'*h);
    Wt2{1} = imfilter(W2, h'*h);
    temp_img1 = W1;
    temp_img2 = W2;
    out1=double(double(img1));
    out2=double(double(img2));
    out_R1{1} = out1(:,:,1);
    temp_img_R1 = out1(:,:,1);
    out_G1{1} = out1(:,:,2);
    temp_img_G1 = out1(:,:,2);
    out_B1{1} = out1(:,:,3);
    temp_img_B1 = out1(:,:,3);
    out_R2{1} = out2(:,:,1);
    temp_img_R2 = out2(:,:,1);
    out_G2{1} = out2(:,:,2);
    temp_img_G2 = out1(:,:,2);
    out_B2{1} = out2(:,:,3);
    temp_img_B2 = out1(:,:,3);
    % Here we are saving the details of all the subsequent level details in
    % different variables for the use in the Laplacian Pyramid block below
    for i = 2 : 5
    temp_img1 = temp_img1(1:2:end,1:2:end);
    Wt1{i} = imfilter(temp_img1, h'*h);
    temp_img2 = temp_img2(1:2:end,1:2:end);
    Wt2{i} = imfilter(temp_img2, h'*h);
    temp_img_R1 = temp_img_R1(1:2:end,1:2:end);
    out_R1{i} = temp_img_R1;
    temp_img_R2 = temp_img_R2(1:2:end,1:2:end);
    out_R2{i} = temp_img_R2;
    temp_img_G1 = temp_img_G1(1:2:end,1:2:end);
    out_G1{i} = temp_img_G1;
    temp_img_G2 = temp_img_G2(1:2:end,1:2:end);
    out_G2{i} = temp_img_G2;
    temp_img_B1 = temp_img_B1(1:2:end,1:2:end);
    out_B1{i} = temp_img_B1;
    temp_img_B2 = temp_img_B2(1:2:end,1:2:end);
    out_B2{i} = temp_img_B2;
    end

    % Computing the size of the pyramid for all the 5 levels of the normalized
    % and equalized image
    % Gaussian Pyramid generation is done by subsequent Blurring and subsampling
    % the original image held in 'out_R1{1}'.Each pixel containing a local average
    % that corresponds to a pixel neighborhood on a lower level of the pyramid. 
    % This technique also used in 'texture synthesis' which is the process of 
    % algorithmically constructing a large digital image from a small digital 
    % sample image by taking advantage of its structural content.

    % A Laplacian pyramid is very similar to a Gaussian pyramid but saves the 
    % difference image of the blurred versions between each levels. Only the 
    % smallest level is not a difference image to enable reconstruction of the 
    % high resolution image using the difference images on higher levels. 
    % This technique can be used in image compression.

    % Here we are doing the Laplacian Pyramid generation by saving each level
    for i = 1 : 4
    [mr1, nr1] = size(out_R1{i});
    [mg1, ng1] = size(out_G1{i});
    [mb1, nb1] = size(out_B1{i});
    [mr2, nr2] = size(out_R2{i});
    [mg2, ng2] = size(out_G2{i});
    [mb2, nb2] = size(out_B2{i});
    out_R1{i} = out_R1{i}-imresize(out_R1{i+1},[mr1,nr1]);
    out_G1{i} = out_G1{i}-imresize(out_G1{i+1},[mg1,ng1]);
    out_B1{i} = out_B1{i}-imresize(out_B1{i+1},[mb1,nb1]);
    out_R2{i} = out_R2{i}-imresize(out_R2{i+1},[mr2,nr2]);
    out_G2{i} = out_G2{i}-imresize(out_G2{i+1},[mg2,ng2]);
    out_B2{i} = out_B2{i}-imresize(out_B2{i+1},[mb2,nb2]);
    end

    % Fusion Process by multiplying different weights with their Normalized and
    % Equalized version. This approach removes the hallows(fringing caused at high 
    % contrast regions) which get generated while fusing inputs with weights.
    for i = 1:5
    R_r{i} = Wt1{i}.*out_R1{i}+Wt2{i}.*out_R2{i};
    R_g{i} = Wt1{i}.*out_G1{i}+Wt2{i}.*out_G2{i};
    R_b{i} = Wt1{i}.*out_B1{i}+Wt2{i}.*out_B2{i};
    end

    % Converting the subsamples at the lowest level back to their original form
    for i = 5:-1:2
    [mr,nr] = size(R_r{i-1});
    [mg,ng] = size(R_g{i-1});
    [mb,nb] = size(R_b{i-1});
    R_r{i-1} = R_r{i-1}+imresize(R_r{i},[mr,nr]);
    R_g{i-1} = R_g{i-1}+imresize(R_g{i},[mg,ng]);
    R_b{i-1} = R_b{i-1}+imresize(R_b{i},[mb,nb]);
    end
    R = R_r{1};
    G = R_g{1};
    B = R_b{1};
    % Fusing the different color components back to get the desired image 
    fusion = cat(3,uint8(R),uint8(G),uint8(B));
    figure
    imshow(img,'InitialMagnification','fit')
    title('Original Image')
    figure
    imshow(fusion,'InitialMagnification','fit')
    title('Modified Image')
end