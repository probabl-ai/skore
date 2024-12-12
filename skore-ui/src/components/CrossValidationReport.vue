<script setup lang="ts">
import CrossValidationReportDetails from "@/components/CrossValidationReportDetails.vue";
import CrossValidationReportPlots from "@/components/CrossValidationReportPlots.vue";
import CrossValidationReportResults from "@/components/CrossValidationReportResults.vue";
import TabPanel from "@/components/TabPanel.vue";
import TabPanelContent from "@/components/TabPanelContent.vue";
import type { DetailSectionDto, PlotDto, ScalarResultDto, TabularResultDto } from "@/dto";

const props = defineProps<{
  scalarResults: ScalarResultDto[];
  tabularResults: TabularResultDto[];
  plots: PlotDto[];
  sections: DetailSectionDto[];
}>();
</script>

<template>
  <div class="cross-validation-report">
    <TabPanel>
      <TabPanelContent name="Primary Results" icon="icon-bar-chart">
        <CrossValidationReportResults
          :scalar-results="props.scalarResults"
          :tabular-results="props.tabularResults"
        />
      </TabPanelContent>
      <TabPanelContent name="Plots" icon="icon-large-bar-chart">
        <CrossValidationReportPlots :plots="props.plots" />
      </TabPanelContent>
      <TabPanelContent name="Storage/Details" icon="icon-hard-drive">
        <CrossValidationReportDetails :sections="props.sections" />
      </TabPanelContent>
    </TabPanel>
  </div>
</template>
