clear;clc;close all
cd C:\Users\DELL\Documents\GitHub\pytorch-deepdream\

all_model = dir('response\*.mat');
example_num=2;
for mm = 1:length(all_model)
    model_now = all_model(mm).name;
    load(['response\', model_now]);
    
    figure;
    interested_channels = [123	2715	1052	3044	2384];
    for channel_now = interested_channels
        heiti_response = all_heiti_data(:,channel_now);
        dp_response = all_dp_data(:,channel_now);
        
        [sort_heiti_response,sort_heiti_idx] = sort(heiti_response);

        [sort_dp_response,sort_dp_idx] = sort(dp_response);

        nexttile([1,3])
        img_to_show=[];
        for ee = 1:example_num
            heiti_img = imread("data\heiti\" + all_heiti_name(sort_heiti_idx(ee),:));
            for ii = 1:3
                heiti_img_3(:,:,ii) = heiti_img;
            end
            img_to_show=[img_to_show,heiti_img_3];
        end

        for ee = 1:example_num
             dp_img = imread("data\out-images\alexnet_" + model_now(1:end-13) + "\" + all_dp_name(sort_dp_idx(ee),:));
            img_to_show=[img_to_show,dp_img];
        end
        imshow(img_to_show)


        nexttile
        histogram(heiti_response)
        xline(sort_dp_response(end+1-ee:end),'LineWidth',2)
        xline(sort_dp_response(1:ee),'LineWidth',2)
        title(['PC ', num2str(channel_now)])
        
        nexttile([1,3])
        img_to_show=[];
        for ee = 1:example_num
            dp_img = imread("data\out-images\alexnet_" + model_now(1:end-13) + "\" + all_dp_name(sort_dp_idx(end+1-ee),:));
            img_to_show=[img_to_show,dp_img];
        end
        for ee = 1:example_num
            heiti_img = imread("data\heiti\" + all_heiti_name(sort_heiti_idx(end+1-ee),:));
            for ii = 1:3
                heiti_img_3(:,:,ii) = heiti_img;
            end
            img_to_show=[img_to_show,heiti_img_3];
        end
        imshow(img_to_show)

    end
    model_now(findstr(model_now,'_'))='-';
    sgtitle(model_now)
    set(gcf,'Position',[  0.1402    0.0634    1.6536    0.7392]*1000)
    saveas(gca,['summary/',model_now,'.png'])
end