<script setup lang="ts">
import Simplebar from "simplebar-vue";
import { computed, ref } from "vue";

import type { Plot } from "@/components/CrossValidationReport.vue";
import DropdownButton from "@/components/DropdownButton.vue";
import DropdownButtonItem from "@/components/DropdownButtonItem.vue";

const props = defineProps<{ plots: Plot[] }>();
const currentPlotIndex = ref(0);
const currentPlot = computed<Plot>(() => {
  return props.plots[currentPlotIndex.value];
});
</script>

<template>
  <div class="cross-validation-report-plots">
    <div class="header">
      <div class="name"><i class="icon icon-bar-chart" /> {{ currentPlot.name }}</div>
      <DropdownButton
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
    <div class="plot"></div>
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
