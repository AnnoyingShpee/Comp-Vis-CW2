function output = get_tiny_images_1(image_paths, image_size)
%GET_TINY_IMAGES 

    dim                             = image_size;
    rows                            = size(image_paths, 1);
    output                          = zeros(rows, dim^2 * 3);
     
    parfor i = 1:rows
        img                         = imread(image_paths{i});
        
        % calc scaling factors
        [img_height, img_width]     = size(img);
        scale_x                     = img_width/dim;
        scale_y                     = img_height/dim;
        
        [x, y]                      = meshgrid(1:dim, 1:dim);
        x_scaled                    = x * scale_x;
        y_scaled                    = y * scale_y;
        
        
        red                         = uint8(interp2(double(img(:,:,1)), x_scaled, y_scaled, 'bilinear'));
        green                       = uint8(interp2(double(img(:,:,1)), x_scaled, y_scaled, 'bilinear'));
        blue                        = uint8(interp2(double(img(:,:,1)), x_scaled, y_scaled, 'bilinear'));
        ref                         = cat(3, red, green, blue);
        
        output(i,:)                 = reshape(ref, 1,[]);
    end
end
