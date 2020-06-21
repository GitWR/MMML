function [sub_t , q_value] = compute_sub(data)
  num_t=length(data);%统计训练图像集个数
  sub_t=cell(1,num_t);
for i=1:num_t
  sample_t=zscore(data{i});%取出每一个图像集,并中心化操作
  cov_t=sample_t*sample_t';
%   cov_t=cov_t+trace(cov_t)*(1e-3)*eye(size(cov_t,1));
  [U,~,~]=svd(cov_t);%奇异值分解
  if num_t < 40
      q_value = 12;
  else
      q_value = 40;
  end
  sub_t{i}=U(:,1:q_value);%取出目标维数下的数据
   %从上面计算来看，线性子空间维数q=40，而论文中的D=400，这两个参数仅针对ETH-80数据集
end
end

