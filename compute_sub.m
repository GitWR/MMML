function [sub_t , q_value] = compute_sub(data)
  num_t=length(data);%ͳ��ѵ��ͼ�񼯸���
  sub_t=cell(1,num_t);
for i=1:num_t
  sample_t=zscore(data{i});%ȡ��ÿһ��ͼ��,�����Ļ�����
  cov_t=sample_t*sample_t';
%   cov_t=cov_t+trace(cov_t)*(1e-3)*eye(size(cov_t,1));
  [U,~,~]=svd(cov_t);%����ֵ�ֽ�
  if num_t < 40
      q_value = 12;
  else
      q_value = 40;
  end
  sub_t{i}=U(:,1:q_value);%ȡ��Ŀ��ά���µ�����
   %��������������������ӿռ�ά��q=40���������е�D=400�����������������ETH-80���ݼ�
end
end

