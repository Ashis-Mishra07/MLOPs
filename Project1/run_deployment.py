from pipeline.deployment_pipeline import deployment_pipeline , inference_pipeline
import click

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    '--mode', 
    default=DEPLOY_AND_PREDICT, 
    help='deploy, predict or deploy_and_predict',
)
@click.option(
    "--min-accuracy",
    default = 0.92 ,
    help = "Minimum accuracy required for deployment" ,
)

def run_deployment(mode, min_accuracy):
    if mode == DEPLOY:
        deployment_pipeline()
    elif mode == PREDICT:
        inference_pipeline()
    elif mode == DEPLOY_AND_PREDICT:
        deployment_pipeline()
        inference_pipeline()
    else:
        print("Invalid mode")

