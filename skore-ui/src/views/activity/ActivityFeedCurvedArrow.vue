<script setup lang="ts">
const props = defineProps<{
  hasArrow: boolean;
}>();
</script>

<template>
  <div class="arrow">
    <svg width="14" height="5" viewBox="0 0 14 5" v-if="props.hasArrow">
      <path
        d="M3.65366 0.20274C3.86579 -0.0109393 4.20971 -0.0109393 4.42184 0.20274L7.87865 3.68484C8.09078 3.89852 8.09078 4.24497 7.87865 4.45864C7.66653 4.67232 7.3226 4.67232 7.11047 4.45864L4.58094 1.9106L4.58094 5.00329H3.49456L3.49456 1.9106L0.965026 4.45864C0.752899 4.67232 0.408972 4.67232 0.196845 4.45864C-0.0152823 4.24497 -0.0152823 3.89852 0.196845 3.68484L3.65366 0.20274Z"
      />
    </svg>
    <svg width="14" height="5" v-else></svg>
    <svg viewBox="0 0 14 4" preserveAspectRatio="none">
      <path d="M3.5 4L3.49445 0.00202452L4.58112 0.00202452L4.58112 4.00202L3.5 4Z" />
    </svg>
    <svg width="14" height="21" viewBox="0 0 14 21">
      <path
        d="M3.9586 10.3368C3.65428 7.65424 3.54634 4.11207 3.49444 0.00202484L4.58203 0.00202484C4.63382 4.10271 4.65366 7.59226 4.95222 10.2241C5.2513 12.8604 5.70708 14.8009 6.37338 16.2185C7.03218 17.6202 7.89364 18.5052 9.02527 19.053C10.1751 19.6095 11.9479 19.9531 13.8636 19.9531L13.8636 20.9531C11.8861 20.9531 9.93739 20.6055 8.58957 19.9531C7.22359 19.2918 6.21042 18.2227 5.46837 16.6439C4.73383 15.0811 4.26239 13.0148 3.9586 10.3368Z"
      />
    </svg>
  </div>
</template>

<style scoped>
.arrow {
  display: flex;
  height: 400px;
  flex-direction: column;
  justify-content: center;

  & svg {
    clip-rule: evenodd;
    fill: var(--color-stroke-background-primary);
    fill-rule: evenodd;

    &:nth-child(2) {
      width: 14px;
      flex: 1;
    }
  }
}
</style>
