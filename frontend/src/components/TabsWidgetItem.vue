<script setup lang="ts">
import { inject } from "vue";

const props = defineProps<{
  value: number;
}>();
const currentTabIndex = inject("currentTabIndex");
</script>

<template>
  <div v-if="props.value == currentTabIndex">
    <slot></slot>
  </div>
</template>
