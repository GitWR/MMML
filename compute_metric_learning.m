function [ U ] = compute_metric_learning( k_Gras, k_Spd, lamda1, lamda2, Train_lables)
  num_class = length(unique(Train_lables)); % 类别数
  D = size(k_Gras,1); % 141
  d = 8; % 需要调节的参数
  itera = 1;
  % U = rand(D,d);
for i=1:itera
  fprintf('\n i= %d \n',i);
  
  Sw=zeros(D,D); % 141 * 141
  Sb=zeros(D,D); % 141 * 141
  Nw=0;
  Nb=0;
  for j = 1 : num_class
      num_eachclass = find(Train_lables==j);%统计出每一类的样本的索引
      for k = 1 : length(num_eachclass)
          K_gras_data1 = k_Gras(num_eachclass(k),:);
          K_spd_data1 = k_Spd(num_eachclass(k),:);
          for m=k+1 : length(num_eachclass)
              K_gras_data2 = k_Gras(num_eachclass(m),:);
              K_spd_data2 = k_Spd(num_eachclass(m),:);
              Sw_temp_gras = lamda1*(K_gras_data1'-K_gras_data2')*(K_gras_data1'-K_gras_data2')'*lamda1;
              Sw_temp_spd = lamda2*(K_spd_data1'-K_spd_data2')*(K_spd_data1'-K_spd_data2')'*lamda2;
              Sw_temp = Sw_temp_gras + Sw_temp_spd;
              Sw = Sw+Sw_temp;
              Nw=Nw+1;%用于统计成对的计算样本的个数
          end
      end
  end
  
  for j=1:num_class
      num_eachclass=find(Train_lables==j);%统计出每一类的样本的索引
      num_difclass=find(Train_lables~=j);%统计出和j不同类的样本索引
      for k=1:length(num_eachclass)
          K_gras_data1 = k_Gras(num_eachclass(k),:);
          K_spd_data1 = k_Spd(num_eachclass(k),:);
          for m=1:length(num_difclass)
              K_gras_data2 = k_Gras(num_difclass(m),:);
              K_spd_data2 = k_Spd(num_difclass(m),:);
              Sb_temp_gras = lamda1*(K_gras_data1'-K_gras_data2')*(K_gras_data1'-K_gras_data2')'*lamda1;
              Sb_temp_spd = lamda2*(K_spd_data1'-K_spd_data2')*(K_spd_data1'-K_spd_data2')'*lamda2;
              Sb_temp = Sb_temp_gras + Sb_temp_spd;
              Sb=Sb+Sb_temp;
              Nb=Nb+1;%用于统计成对的计算样本的个数
          end
      end
  end
  
  Sw_final=Sw/(Nw);%论文中的公式（9）,400*400
  Sb_final=Sb/(Nb);%论文中的公式（10）,400*400
  
%这里的迭代终止条件怎么给，我们希望投影后类内尽量靠的近，类间尽量散得开，因此每迭代一次，就分的越开。
  %Object_function = Sw_final-alpha*Sb_final;
  % if (i>1)
  %     if (Cost(i)>=Cost(i-1))
  %         count1=count1+1;
  %         if (count1>=1)                  （收敛性说明，因此屏蔽判定条件）
  %            break;
  %         end
  %     end
  % end
  % 优化
  [Object_V , Object_E] = eig(inv(Sw_final)*Sb_final); % 取最大值
  E_unsort = diag(Object_E);
  [~ , index] = sort(E_unsort,'descend'); % 特征值降序排列
  V_sort = Object_V(:,index);%按照特征值的顺序，将特征向量降序排列
  U = V_sort(:,1:d);
  
  Cost(i) = det(U'*Sb_final*U)/det(U'*Sw_final*U);
  fprintf(' iter\t            cost val\t \n    ');
  fprintf('%5d\t \n%+.16e\t \n', i, Cost(:,1:i));
  
end
end

