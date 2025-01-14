library(tidyverse)
library(ggplot2)

setwd('C:/git/hpo_ctpe_forsharing')

#materials = c('ABS', 'PPS', 'GFRE')
materials = c('ABS')
h4 <- hcl.colors(10, palette = "Fall")
cm_to_inch = 0.393701

# debonding strength vs cost graph ----------------------------------------------

dat <- map(materials, ~read_xlsx(paste0('./results/adhesive_bonding_manyeval_material_', ., '.csv'))) %>% 
  bind_rows(.id = 'material') %>% 
  mutate(plasma_distance = dplyr::coalesce(plasma_distance, plasma_distance_value),
         plasma_passes = dplyr::coalesce(plasma_passes, plasma_passes_value),
         plasma_power = dplyr::coalesce(plasma_power, plasma_power_value),
         plasma_speed = dplyr::coalesce(plasma_speed, plasma_speed_value)) %>% 
  select(!ends_with('value'))

dat$material <- factor(materials[as.numeric(dat$material)])

dat %>% 
  group_by(material) %>% 
  filter(VisualQ == 1) %>% 
  tally()/20000

dat <- dat %>% 
  filter(VisualQ == 1)

optim_dat <- map(materials, ~read_xlsx(paste0('./adhesive_bonding_optimal_parameters_material', ., '.csv'))) %>% 
  bind_rows(.id = 'material') %>% 
  mutate(plasma_distance = dplyr::coalesce(plasma_distance, plasma_distance_value),
         plasma_passes = dplyr::coalesce(plasma_passes, plasma_passes_value),
         plasma_power = dplyr::coalesce(plasma_power, plasma_power_value),
         plasma_speed = dplyr::coalesce(plasma_speed, plasma_speed_value)) %>% 
  select(!ends_with('value'))
optim_dat$material <- materials[as.numeric(optim_dat$material)]

# all optim values have VisualQ ==2?

optim_dat %>% 
  group_by(material) %>% 
  filter(VisualQ == 2) %>% 
  tally()/30

dat$type <-  'outcome space'
optim_dat$type <-  'TPE-optimised'

dat_eval_optim <- bind_rows(dat, optim_dat)
dat_eval_optim$material <- fct_relevel(dat_eval_optim$material, c('ABS', 'PPS', 'GFRE'))

p1 <- dat_eval_optim %>%
  filter(material != 'Aluminum') %>%
  ggplot(aes(x = cost, y = tensileStrength, colour = type, size = type, shape = as.character(VisualQ))) +
  geom_point() +
  scale_colour_manual(values = c('outcome space' = alpha(h4[1], 0.6),
                                 'TPE-optimised' = h4[10])) +
  scale_size_manual(values = c('outcome space' = 1, 
                               'TPE-optimised' = 2)) +
  facet_wrap(~ material, ncol = 3) +
  theme_minimal() +
  theme(panel.spacing = unit(3, "lines"),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 13),
        strip.text = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        plot.title = element_text(size = 16, face = "bold")) +
  labs(x = "Cost", y = "Bonding strength", title = "The outcome space (20 000 random evaluations, but only VisualQ=OK kept) and TPE-optimised values (30 replications) of bonding strength and cost")

# Save the updated plot
ggsave(filename = './strength_vs_cost_plots.png',
       plot = p1,
       device = 'png',
       units = "in",
       dpi = 300,
       height = 20 * cm_to_inch,
       width = 45 * cm_to_inch,
       bg = 'white'
)

# trajectory graph --------------------------------------------------------------

dat <- map(materials, ~read.csv(paste0('./results/adhesive_bonding_material', ., '_intermediate_results.csv'))) %>% 
  bind_rows(.id = 'material')
dat$material <- factor(materials[as.numeric(dat$material)])
dat$material <- fct_relevel(dat$material, c('ABS', 'PPS', 'GFRE'))

degf <- length(unique(dat$macrorep))

dat2 <- dat %>% 
  mutate(best_loss = cumulative_max_loss) %>% 
  group_by(material, iteration) %>% 
  summarise(mean_loss = mean(best_loss),
            sd_loss = sd(best_loss), 
            sd_loss_div = sd(best_loss)/sqrt(degf), 
            lb = mean(best_loss) - qt(0.975, df = degf-1) * sd_loss_div,
            ub = mean(best_loss) + qt(0.975, df = degf-1) * sd_loss_div)

p2 <- ggplot(dat2, aes(x = iteration, y = mean_loss)) +
  geom_line() +
  geom_ribbon(aes(ymin = lb, ymax = ub), alpha = 0.2) +
  facet_wrap(~ material, ncol = 3) +
  labs(x = "Iteration", y = "Bonding strength", title = "Mean bonding strength vs iteration across 30 replications") +
  theme_minimal() +
  theme(panel.spacing = unit(2, "lines"),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 14),
        plot.title = element_text(size = 16, face = "bold"))

ggsave(filename = './results/optimisation_trajectory.png',
       plot = p2,
       device = 'png',
       units = "in",
       dpi = 300,
       height = 20 * cm_to_inch,
       width = 45 * cm_to_inch,
       bg = 'white'
)

# optimum iteration number ------------------------------------------------

dat_max_iter <- dat %>% distinct(material, macrorep, best_row_number)

p3 <- dat_max_iter %>% 
  ggplot(aes(x = material, y= best_row_number)) +
  geom_boxplot() +
  labs(y = "Iteration index", title = "Iteration number where optimum was found (across 30 replications)")+
  theme_minimal() +
  theme(panel.spacing = unit(2, "lines"),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        strip.text = element_text(size = 14),
        plot.title = element_text(size = 16, face = "bold"))

ggsave(filename = './results/index_optimum.png',
       plot = p3,
       device = 'png',
       units = "in",
       dpi = 300,
       height = 20 * cm_to_inch,
       width = 45 * cm_to_inch,
       bg = 'white'
)