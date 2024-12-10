<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, ref } from "vue";

import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";
import PlotlyWidget from "@/components/PlotlyWidget.vue";
import type { PlotDto } from "@/dto";

const props = defineProps<{ plots: PlotDto[] }>();
const currentPlotIndex = ref(0);
const currentPlot = computed<PlotDto>(() => {
  return props.plots[currentPlotIndex.value];
});
</script>

<template>
  <div class="cross-validation-report-plots" v-if="currentPlot">
    <div class="header">
      <div class="name"><i class="icon icon-bar-chart" /> {{ currentPlot.name }}</div>
      <DropdownButton
        v-if="props.plots.length > 1"
        icon="icon-chevron-down"
        align="left"
        icon-position="right"
        :label="currentPlot.name"
      >
        <Simplebar>
          <DropdownButtonItem
            v-for="(result, i) in props.plots"
            :key="i"
            :label="result.name"
            @click="currentPlotIndex = i"
          />
        </Simplebar>
      </DropdownButton>
    </div>
    <div class="plot">
      <PlotlyWidget :spec="currentPlot.value" />
    </div>
  </div>
</template>

<style>
.cross-validation-report-plots {
  & .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--spacing-16) var(--spacing-10);

    & .name {
      color: var(--color-text-primary);
      font-size: var(--font-size-sm);

      & .icon {
        color: var(--color-text-branding);
        vertical-align: middle;
      }
    }

    & .dropdown {
      & button {
        padding: var(--spacing-6) var(--spacing-10);
      }
    }
  }
}
</style>
