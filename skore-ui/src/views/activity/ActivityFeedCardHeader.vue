<script setup lang="ts">
import { format, parseISO } from "date-fns";
import { computed } from "vue";

const props = defineProps<{
  icon: string;
  name: string;
  datetime: string;
}>();

const datetime = computed(() => parseISO(props.datetime));
const date = computed(() => format(datetime.value, "yyyy/MM/dd"));
const time = computed(() => format(datetime.value, "HH:mm:ss"));
</script>

<template>
  <div class="activity-feed-card-header">
    <div class="name"><i class="icon" :class="props.icon" /> {{ props.name }}</div>
    <div class="datetime">{{ date }} <span class="at">at</span> {{ time }}</div>
  </div>
</template>

<style scoped>
.activity-feed-card-header {
  display: flex;
  flex-direction: row;
  justify-content: space-between;

  & .name {
    color: var(--color-text-primary);
    font-size: var(--font-size-xs);

    & .icon {
      color: var(--color-icon-tertiary);
      vertical-align: middle;
    }
  }

  & .datetime {
    color: var(--color-text-primary);
    font-size: var(--font-size-xs);

    & .at {
      color: var(--color-text-secondary);
    }
  }
}
</style>
