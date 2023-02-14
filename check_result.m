clear;clc;close all
cd C:\Users\DELL\Documents\GitHub\pytorch-deepdream\

all_model = dir('response\*.mat');
example_num=6;
for mm = 1:length(all_model)
    model_now = all_model(mm).name;
    load(['response\', model_now]);
    
    %% do for fc7(pc1234)
    figure;
    interested_channels = [1,2,3,4];
    for channel_now = interested_channels
        heiti_response = heiti_response_fc7(:,channel_now);
        dp_response = dp_fc7_response(:,channel_now);
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
             dp_img = imresize(imread("data\out-images\alexnet_" + model_now(1:end-13) + "_fc7\" + all_dp_fc7_name(sort_dp_idx(ee),:)),[224,224]);
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
            dp_img = imresize(imread("data\out-images\alexnet_" + model_now(1:end-13) + "_fc7\" + all_dp_fc7_name(sort_dp_idx(end+1-ee),:)),[224,224]);
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
    temp_name = model_now;
    temp_name(findstr(temp_name,'_'))='-';
    sgtitle(temp_name)
    set(gcf,'Position',[  0.1402    0.0634    1.6536    0.7392]*1000)
    saveas(gca,['summary/',model_now,'.png'])


    %% do for fc6(word-selective unit)
    figure;
    dprime_all = 1+load(['selectivity\', model_now(1:end-16), 'dprime_idx_fc6.mat']).dprime_idx;
    interested_channels = [dprime_all(end-4:end), dprime_all(1)]
    for channel_now = interested_channels
        heiti_response = heiti_response_fc6(:,channel_now);
        dp_response = dp_fc6_response(:,channel_now);
        [sort_heiti_response,sort_heiti_idx] = sort(heiti_response);
        [sort_dp_response,sort_dp_idx] = sort(dp_response);

        nexttile([1,2])
        histogram(heiti_response)
        xline(sort_dp_response(end+1-ee:end),'LineWidth',2)

        title(['Unit ', num2str(channel_now)])
        if(channel_now==interested_channels(end))
            title('most non-selective unit')
        end
        nexttile([1,3])
        
        img_to_show=[];
        for ee = 1:example_num
            dp_img = imresize(imread("data\out-images\alexnet_" + model_now(1:end-13) + "_fc6\" + all_dp_fc6_name(sort_dp_idx(end+1-ee),:)),[224,224]);
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
        title(all_dp_fc6_name(sort_dp_idx(end+1-ee),:),Interpreter="none")

    end
    model_now(findstr(model_now,'_'))='-';
    sgtitle(model_now)
    set(gcf,'Position',[   0.2818   -0.0230    1.3184    1.0808]*1000)
    saveas(gca,['summary/',model_now,'fc6.png'])
end