# SageMaker notebookでendpointを作成
import sagemaker
from sagemaker.estimator import Estimator

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
print(f"sagemaker role arn: {role}")
print(f"sagemaker session region: {sess.boto_region_name}")


estimator = Estimator(
    image_uri="aws_image_uri",
    role=sagemaker.get_execution_role(),
    instance_type="ml.g4dn.2xlarge",
    instance_count=1,
)
estimator.fit()
# endpoint deploy
predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.g4dn.2xlarge')

result = predictor.predict(data='For LTE TM5-10 and NR, no special instructions are needed because the precoding may be included in a digital beamforming processing block within the O-RU. In the UL, OFDM phase compensation (for all channels except PRACH) [3GPP TS38.211 clause 5.4], FFT, CP removal andDigital beamforming functions reside in the O.RU, while the rest of the PHY functions including resource element de-mapping, equalization, de-modulation, rate de-matching and de-coding reside in O-DU. If the same interface and split point is used for DL and UL, specification effort can be reduced.')
# delete endpoint
sagemaker.predictor.Predictor.delete_endpoint(predictor, delete_endpoint_config=True)
