## Eagle Image Classification for Hawk Watch International (HWI)

[Hawk Watch International](https://hawkwatch.org/) is a nonprofit organization with the mission to conserve our environment through education, long-term monitoring, and scientific research on raptors as indicators of ecosystem health.

In 2018, Hawk Watch had a grant to measure the distance roadkill deer need to be from the road to prevent feeding eagles from being hit by cars. Hawk Watch placed game cameras near roadkill deer to monitor the behavior of raptors feeding on the deer as cars drove by and spooked the raptors. Hawk Watch gathered more than a million images with less than 5% containing raptors. They had three months to review the images to find the raptor images then do their analysis of the birds' behavior. They also had no budget to pay for image recognition in the cloud using TensorFlow. This meant any possible solution would need to run on a standard laptop.

My goal was to see if it was possible to use older laptop-friendly image recogition machine learning to filter the images. We wondered if the older image recognition would be sufficient the images contained a limited number of objects (a dead deer, birds, clouds, plants and the occasional coyote). The answer was 'No, a laptop is not sufficient.' The image recognition needed to be done in the cloud. Given no budget, the work was crowd sourced with volunteer manually reviewing images.

This repo contains the code with my attempt to make the image recognition work on a laptop.

## Slide Deck showing approach and results
[Slide-Deck](https://github.com/Rebeccachristman/Presentations/blob/main/HawkWatch-EagleImageClassification.pdf)

## Process Diagram
![Diagram](https://github.com/Rebeccachristman/HawkWatch/blob/main/doc/HWIImageClassificationDiagram.jpeg)