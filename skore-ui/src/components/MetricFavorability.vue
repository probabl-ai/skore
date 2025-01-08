<script setup lang="ts">
import type { Favorability } from "@/dto";
import { computed } from "vue";

import FloatingTooltip from "@/components/FloatingTooltip.vue";

const props = defineProps<{ favorability: Favorability }>();

const icon = computed(() => {
  switch (props.favorability) {
    case "greater_is_better":
      return "icon-ascending-arrow";
    case "lower_is_better":
      return "icon-descending-arrow";
    default:
      return "";
  }
});

const text = computed(() => {
  switch (props.favorability) {
    case "greater_is_better":
      return "Higher is better";
    case "lower_is_better":
      return "Lower is better";
    default:
      return "";
  }
});
</script>

<template>
  <FloatingTooltip v-if="icon !== ''" placement="bottom" :text="text">
    <i class="icon" :class="icon" />
  </FloatingTooltip>
</template>

<style scoped>
.icon {
  color: var(--color-icon-tertiary);
  vertical-align: middle;
}
</style>
