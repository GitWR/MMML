function [ U ] = compute_metric_learning( k_Gras, k_Spd, lamda1, lamda2, Train_lables)
  num_class = length(unique(Train_lables)); % �����
  D = size(k_Gras,1); % 141
  d = 8; % ��Ҫ���ڵĲ���
  itera = 1;
  % U = rand(D,d);
for i=1:itera
  fprintf('\n i= %d \n',i);
  
  Sw=zeros(D,D); % 141 * 141
  Sb=zeros(D,D); % 141 * 141
  Nw=0;
  Nb=0;
  for j = 1 : num_class
      num_eachclass = find(Train_lables==j);%ͳ�Ƴ�ÿһ�������������
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
              Nw=Nw+1;%����ͳ�ƳɶԵļ��������ĸ���
          end
      end
  end
  
  for j=1:num_class
      num_eachclass=find(Train_lables==j);%ͳ�Ƴ�ÿһ�������������
      num_difclass=find(Train_lables~=j);%ͳ�Ƴ���j��ͬ�����������
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
              Nb=Nb+1;%����ͳ�ƳɶԵļ��������ĸ���
          end
      end
  end
  
  Sw_final=Sw/(Nw);%�����еĹ�ʽ��9��,400*400
  Sb_final=Sb/(Nb);%�����еĹ�ʽ��10��,400*400
  
%����ĵ�����ֹ������ô��������ϣ��ͶӰ�����ھ������Ľ�����価��ɢ�ÿ������ÿ����һ�Σ��ͷֵ�Խ����
  %Object_function = Sw_final-alpha*Sb_final;
  % if (i>1)
  %     if (Cost(i)>=Cost(i-1))
  %         count1=count1+1;
  %         if (count1>=1)                  ��������˵������������ж�������
  %            break;
  %         end
  %     end
  % end
  % �Ż�
  [Object_V , Object_E] = eig(inv(Sw_final)*Sb_final); % ȡ���ֵ
  E_unsort = diag(Object_E);
  [~ , index] = sort(E_unsort,'descend'); % ����ֵ��������
  V_sort = Object_V(:,index);%��������ֵ��˳�򣬽�����������������
  U = V_sort(:,1:d);
  
  Cost(i) = det(U'*Sb_final*U)/det(U'*Sw_final*U);
  fprintf(' iter\t            cost val\t \n    ');
  fprintf('%5d\t \n%+.16e\t \n', i, Cost(:,1:i));
  
end
end

